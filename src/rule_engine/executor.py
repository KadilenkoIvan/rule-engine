"""Исполнение плана: проход по рёбрам, задержки, ошибки, повторы и откат.

Протокол прохода по ребру ``source --action--> target``:

1. GUARD   — по убеждению движка устройство в ``source`` и условие ребра выполнено;
2. READ    — читаем реальное состояние устройства:
              * уже ``target`` → шаг выполнен кем-то ещё (идемпотентно, без команды);
              * не ``source``  → STATE_DRIFT: синхронизируем мир и перестраиваем путь (DP);
3. SEND    — команда с таймаутом ``min(timeout шага, остаток дедлайна)``;
4. VERIFY  — после успеха читаем состояние и сверяем с ``target``;
              после TIMEOUT тоже читаем: команда могла примениться с опозданием,
              и слепой повтор был бы ошибкой (неизвестный исход ≠ неудача);
5. COMMIT  — переход в автомате мира + запись в журнал компенсаций.

Временные ошибки (TIMEOUT, DEVICE_ERROR, VERIFY_FAILED) повторяются с
экспоненциальным backoff в пределах числа попыток, выделенного DP-бюджетом.
Неудача критичного шага откатывает уже выполненные шаги в обратном порядке
(сага); путь отката по автомату ищется тем же DP, значения восстанавливаются.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from rule_engine.errors import EngineError, ErrorCode
from rule_engine.fsm import plan_path
from rule_engine.planner import Plan, Step, allocate_attempts
from rule_engine.rules import RuleSet
from rule_engine.world import World


class VirtualClock:
    """Виртуальное время: исполнение детерминировано и тестируемо."""

    def __init__(self, start_ms: float = 0.0) -> None:
        self.now_ms = start_ms

    def __call__(self) -> float:
        return self.now_ms

    def advance(self, ms: float) -> None:
        self.now_ms += max(0.0, ms)

    sleep = advance


class AttemptStatus(StrEnum):
    OK = "ok"
    TIMEOUT = "timeout"
    ERROR = "error"  # временная ошибка
    FAULT = "fault"  # неисправность, повтор бесполезен
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class Attempt:
    status: AttemptStatus
    elapsed_ms: float
    detail: str = ""


class Driver(Protocol):
    def send(self, step: Step, timeout_ms: float) -> Attempt: ...

    def read_state(self, device: str) -> tuple[str, dict[str, float]] | None: ...


class SimulatedDriver:
    """Симулятор устройств: задержки, сбои, поздние применения, внешние изменения.

    Хранит *физическое* состояние отдельно от мира движка. Поддерживает
    инъекцию отказов для тестов и демонстраций: ``inject("tv", "timeout", 2)``.
    """

    def __init__(self, ruleset: RuleSet, world: World, clock: VirtualClock, seed: int = 0) -> None:
        self.ruleset = ruleset
        self.clock = clock
        self.rng = random.Random(seed)
        self.read_latency_ms = float(ruleset.spec.execution.get("read_latency_ms", 30))
        self.physical: dict[str, tuple[str, dict[str, float]]] = {
            d: (s.state, dict(s.values)) for d, s in world.devices.items()
        }
        self.online = {d: s.online for d, s in world.devices.items()}
        self.faults: dict[str, list[str]] = {}
        self.sent: list[tuple[float, str, str]] = []

    def inject(self, device: str, kind: str, times: int = 1) -> None:
        """kind: timeout | late (таймаут, но применилось) | error | fault | offline | drift:<state>"""
        if kind == "offline":
            self.online[device] = False
            return
        if kind == "online":
            self.online[device] = True
            return
        if kind.startswith("drift:"):
            state = kind.split(":", 1)[1]
            _, values = self.physical[device]
            self.physical[device] = (state, values)
            return
        self.faults.setdefault(device, []).extend([kind] * times)

    def _latency(self, step: Step) -> float:
        device = self.ruleset.spec.devices[step.device]
        base = device.sim.latency_ms or step.latency_ms
        return base * math.exp(self.rng.gauss(0.0, 0.25))

    def _apply(self, step: Step) -> None:
        state, values = self.physical[step.device]
        values = dict(values)
        if step.prop is not None and step.value is not None:
            values[step.prop] = step.value
        self.physical[step.device] = (step.target, values)

    def send(self, step: Step, timeout_ms: float) -> Attempt:
        self.sent.append((self.clock(), step.device, step.action))
        if not self.online.get(step.device, True):
            self.clock.advance(min(timeout_ms, 50.0))
            return Attempt(AttemptStatus.OFFLINE, min(timeout_ms, 50.0), "нет связи")
        injected = self.faults.get(step.device)
        fault = injected.pop(0) if injected else None
        forced = fault is not None
        sim = self.ruleset.spec.devices[step.device].sim
        if fault is None and self.rng.random() < sim.failure_rate:
            fault = "timeout" if self.rng.random() < 0.5 else "error"
        physical_state = self.physical[step.device][0]
        if fault in ("timeout", "late"):
            self.clock.advance(timeout_ms)
            late = fault == "late" or (
                fault == "timeout" and not forced and self.rng.random() < sim.late_apply_rate
            )
            if late and physical_state == step.source:
                self._apply(step)
            return Attempt(AttemptStatus.TIMEOUT, timeout_ms, "ответ не получен")
        latency = self._latency(step)
        if fault == "error":
            self.clock.advance(latency / 2)
            return Attempt(AttemptStatus.ERROR, latency / 2, "устройство вернуло ошибку")
        if fault == "fault":
            self.clock.advance(latency / 2)
            return Attempt(AttemptStatus.FAULT, latency / 2, "аппаратная неисправность")
        if latency > timeout_ms:
            self.clock.advance(timeout_ms)
            if physical_state == step.source:
                self._apply(step)  # устройство медленное, но команду выполнило
            return Attempt(AttemptStatus.TIMEOUT, timeout_ms, "медленный ответ")
        self.clock.advance(latency)
        if physical_state != step.source and physical_state != step.target:
            return Attempt(AttemptStatus.ERROR, latency, f"устройство в «{physical_state}», команда отклонена")
        self._apply(step)
        return Attempt(AttemptStatus.OK, latency)

    def read_state(self, device: str) -> tuple[str, dict[str, float]] | None:
        self.clock.advance(self.read_latency_ms)
        if not self.online.get(device, True):
            return None
        state, values = self.physical[device]
        return state, dict(values)


class ReliabilityModel:
    """Онлайн-оценка надёжности устройств: Beta-апостериорное среднее успеха попытки."""

    def __init__(self, ruleset: RuleSet, prior_strength: float = 10.0) -> None:
        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}
        for device in ruleset.spec.devices.values():
            fail = min(max(device.sim.failure_rate, 0.01), 0.99)
            self.alpha[device.id] = (1 - fail) * prior_strength
            self.beta[device.id] = fail * prior_strength

    def p_success(self, device: str) -> float:
        a, b = self.alpha.get(device, 9.0), self.beta.get(device, 1.0)
        return a / (a + b)

    def update(self, device: str, ok: bool) -> None:
        if ok:
            self.alpha[device] = self.alpha.get(device, 9.0) + 1
        else:
            self.beta[device] = self.beta.get(device, 1.0) + 1


class StepStatus(StrEnum):
    DONE = "done"
    ALREADY = "already"  # устройство уже было в целевом состоянии
    SKIPPED = "skipped"  # некритичный шаг не выполнен
    FAILED = "failed"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


class ExecStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIALLY_SUCCEEDED"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"


@dataclass
class StepLog:
    step: Step
    status: StepStatus
    attempts: int = 0
    elapsed_ms: float = 0.0
    errors: list[EngineError] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


@dataclass
class JournalEntry:
    step: Step
    before_state: str
    before_value: float | None


@dataclass
class ExecutionReport:
    status: ExecStatus
    logs: list[StepLog] = field(default_factory=list)
    rollback: list[StepLog] = field(default_factory=list)
    errors: list[EngineError] = field(default_factory=list)
    journal: list[JournalEntry] = field(default_factory=list)
    elapsed_ms: float = 0.0
    success_probability: float = 1.0

    def done_steps(self) -> list[StepLog]:
        return [log for log in self.logs if log.status in (StepStatus.DONE, StepStatus.ALREADY)]


@dataclass
class ExecutionConfig:
    deadline_ms: float = 8000.0
    max_attempts: int = 4
    backoff_ms: float = 100.0
    backoff_factor: float = 2.0
    jitter: float = 0.2
    rollback_attempts: int = 3
    rollback_deadline_ms: float = 6000.0
    max_replans: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class _Deadline(Exception):
    pass


class Executor:
    def __init__(
        self,
        ruleset: RuleSet,
        world: World,
        driver: Driver,
        clock: VirtualClock,
        reliability: ReliabilityModel | None = None,
        config: ExecutionConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.ruleset = ruleset
        self.world = world
        self.driver = driver
        self.clock = clock
        self.reliability = reliability or ReliabilityModel(ruleset)
        self.config = config or ExecutionConfig()
        self.rng = random.Random(seed)

    # ----------------------------------------------------------------- public
    def execute(self, plan: Plan) -> ExecutionReport:
        start = self.clock()
        deadline = start + self.config.deadline_ms
        attempts, p_all = allocate_attempts(
            plan.steps,
            [self.reliability.p_success(s.device) for s in plan.steps],
            self.config.deadline_ms,
            self.config.max_attempts,
            self.config.backoff_ms,
            self.config.backoff_factor,
        )
        for step, k in zip(plan.steps, attempts, strict=True):
            step.attempts = k
        report = ExecutionReport(ExecStatus.SUCCEEDED, success_probability=p_all)
        failed_groups: set[int] = set()
        for step in plan.steps:
            if step.group in failed_groups:
                # предыдущий шаг этой же команды/шага сценария не прошёл
                report.logs.append(StepLog(step, StepStatus.SKIPPED, trace=["пропущен: не выполнен предыдущий шаг"]))
                continue
            log = self._walk(step, deadline, step.attempts, report.journal)
            report.logs.append(log)
            if log.status in (StepStatus.DONE, StepStatus.ALREADY):
                continue
            report.errors += log.errors
            if not step.critical:
                log.status = StepStatus.SKIPPED
                failed_groups.add(step.group)
                report.status = ExecStatus.PARTIAL
                continue
            log.status = StepStatus.FAILED
            report.rollback = self.compensate(report.journal)
            failed = [r for r in report.rollback if r.status is StepStatus.COMPENSATION_FAILED]
            report.status = ExecStatus.ROLLBACK_FAILED if failed else ExecStatus.ROLLED_BACK
            for r in failed:
                report.errors += r.errors
            break
        report.elapsed_ms = self.clock() - start
        return report

    def compensate(self, journal: list[JournalEntry]) -> list[StepLog]:
        """Откат журнала в обратном порядке. Путь назад по автомату — DP."""
        logs: list[StepLog] = []
        deadline = self.clock() + self.config.rollback_deadline_ms
        for entry in reversed(journal):
            step = entry.step
            if step.edge.meta.get("irreversible"):
                logs.append(
                    StepLog(
                        step, StepStatus.COMPENSATION_FAILED,
                        errors=[EngineError(ErrorCode.COMPENSATION_UNAVAILABLE, step.describe())],
                    )
                )
                continue
            comp_steps = self._compensation_steps(entry)
            if comp_steps is None:
                logs.append(
                    StepLog(
                        step, StepStatus.COMPENSATION_FAILED,
                        errors=[EngineError(
                            ErrorCode.COMPENSATION_FAILED,
                            f"{step.device}: нет пути из «{self.world.state(step.device)}» в «{entry.before_state}»",
                        )],
                    )
                )
                continue
            for comp in comp_steps:
                log = self._walk(comp, deadline, self.config.rollback_attempts, None, allow_replan=False)
                log.status = (
                    StepStatus.COMPENSATED
                    if log.status in (StepStatus.DONE, StepStatus.ALREADY)
                    else StepStatus.COMPENSATION_FAILED
                )
                if log.status is StepStatus.COMPENSATION_FAILED:
                    log.errors.append(EngineError(ErrorCode.COMPENSATION_FAILED, comp.describe()))
                logs.append(log)
                if log.status is StepStatus.COMPENSATION_FAILED:
                    break
        return logs

    # -------------------------------------------------------------- internals
    def _compensation_steps(self, entry: JournalEntry) -> list[Step] | None:
        """Шаги отката записи журнала; ``None`` — пути назад нет."""
        step = entry.step
        fsm = self.ruleset.fsms[self.world.type_of(step.device)]
        current = self.world.state(step.device)
        steps: list[Step] = []
        if current != entry.before_state:
            context = self.world.guard_context(step.device, step.obj)
            path = plan_path(fsm, current, lambda s: s == entry.before_state, context)
            if path is None:
                return None
            for edge in path:
                steps.append(
                    Step(
                        step.device, edge.event, edge.source, edge.target, edge, obj=step.obj,
                        critical=True, reason=f"откат {step.describe()}",
                        timeout_ms=step.timeout_ms, latency_ms=step.latency_ms,
                    )
                )
        if step.prop and entry.before_value is not None:
            restore = [
                e for e in fsm.find(entry.before_state, step.action) if e.target == entry.before_state
            ]
            if restore:
                steps.append(
                    Step(
                        step.device, step.action, entry.before_state, entry.before_state, restore[0],
                        value=entry.before_value, prop=step.prop, critical=True,
                        reason=f"вернуть {step.prop}={entry.before_value:g}",
                        timeout_ms=step.timeout_ms, latency_ms=step.latency_ms,
                    )
                )
        return steps

    def _walk(
        self,
        step: Step,
        deadline: float,
        attempts: int,
        journal: list[JournalEntry] | None,
        allow_replan: bool = True,
        replans: int = 0,
    ) -> StepLog:
        log = StepLog(step, StepStatus.FAILED)
        started = self.clock()
        world = self.world
        device = step.device
        try:
            # 1. GUARD — убеждение движка
            context = world.guard_context(device, step.obj)
            if world.state(device) != step.source:
                log.trace.append(f"guard: мир считает «{world.state(device)}», ожидалось «{step.source}»")
                return self._replan(step, log, deadline, attempts, journal, allow_replan, replans, world.state(device))
            if not step.edge.allowed(context):
                log.errors.append(EngineError(ErrorCode.PRECONDITION_FAILED, f"{step.edge.guard_name}"))
                return log
            # 2. READ — реальность
            actual = self._read(device, log, deadline)
            if actual is None:
                return log
            state, values = actual
            if self._reached(step, state, values):
                # Сделано не нами — синхронизируем мир, но в журнал отката не пишем.
                world.sync(device, state, values, "read: уже в целевом состоянии")
                log.trace.append(f"read: {device} уже «{state}», команда не нужна")
                log.status = StepStatus.ALREADY
                return log
            if state != step.source:
                log.errors.append(EngineError(ErrorCode.STATE_DRIFT, f"{device}: «{state}» вместо «{step.source}»"))
                world.sync(device, state, values, "drift")
                return self._replan(step, log, deadline, attempts, journal, allow_replan, replans, state)
            # 3. SEND + 4. VERIFY, с повторами
            for attempt in range(1, attempts + 1):
                log.attempts = attempt
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise _Deadline
                result = self.driver.send(step, min(step.timeout_ms, remaining))
                log.trace.append(f"попытка {attempt}: {result.status.value} за {result.elapsed_ms:.0f} мс")
                if result.status in (AttemptStatus.OK, AttemptStatus.TIMEOUT):
                    actual = self._read(device, log, deadline)
                    if actual is None:
                        return log
                    state, values = actual
                    if self._reached(step, state, values):
                        if result.status is AttemptStatus.TIMEOUT:
                            log.trace.append("таймаут, но чтение показало: команда применилась")
                        self.reliability.update(device, True)
                        self._commit(step, journal, log)
                        log.status = StepStatus.DONE
                        return log
                    code = ErrorCode.TIMEOUT if result.status is AttemptStatus.TIMEOUT else ErrorCode.VERIFY_FAILED
                    log.errors.append(EngineError(code, f"{device}, попытка {attempt}"))
                    if state not in (step.source, step.target):
                        world.sync(device, state, values, "drift")
                        return self._replan(step, log, deadline, attempts - attempt, journal, allow_replan, replans, state)
                elif result.status is AttemptStatus.ERROR:
                    log.errors.append(EngineError(ErrorCode.DEVICE_ERROR, f"{device}: {result.detail}"))
                elif result.status is AttemptStatus.FAULT:
                    log.errors.append(EngineError(ErrorCode.DEVICE_FAULT, f"{device}: {result.detail}"))
                    self.reliability.update(device, False)
                    return log
                else:
                    log.errors.append(EngineError(ErrorCode.DEVICE_OFFLINE, device))
                    world.set_online(device, False)
                    return log
                self.reliability.update(device, False)
                if attempt < attempts:
                    pause = self.config.backoff_ms * self.config.backoff_factor ** (attempt - 1)
                    pause *= 1 + self.rng.uniform(-self.config.jitter, self.config.jitter)
                    if self.clock() + pause >= deadline:
                        raise _Deadline
                    log.trace.append(f"backoff {pause:.0f} мс")
                    self.clock.sleep(pause)
            return log
        except _Deadline:
            log.errors.append(EngineError(ErrorCode.DEADLINE_EXCEEDED, step.describe()))
            return log
        finally:
            log.elapsed_ms = self.clock() - started

    def _read(self, device: str, log: StepLog, deadline: float) -> tuple[str, dict[str, float]] | None:
        if self.clock() >= deadline:
            raise _Deadline
        actual = self.driver.read_state(device)
        if actual is None:
            log.errors.append(EngineError(ErrorCode.DEVICE_OFFLINE, device))
            self.world.set_online(device, False)
        return actual

    @staticmethod
    def _reached(step: Step, state: str, values: dict[str, float]) -> bool:
        if state != step.target:
            return False
        if step.prop is not None and step.value is not None:
            return values.get(step.prop) == step.value
        return step.source != step.target

    def _commit(self, step: Step, journal: list[JournalEntry] | None, log: StepLog, note: str = "") -> None:
        before_value = self.world.value(step.device, step.prop) if step.prop else None
        before_state = self.world.state(step.device)
        self.world.apply(step.device, step.action, step.prop, step.value, step.obj, note or step.reason)
        if journal is not None:
            journal.append(JournalEntry(step, before_state, before_value))
        log.trace.append(f"commit: {step.describe()}" + (f" ({note})" if note else ""))

    def _replan(
        self,
        step: Step,
        log: StepLog,
        deadline: float,
        attempts: int,
        journal: list[JournalEntry] | None,
        allow_replan: bool,
        replans: int,
        state: str,
    ) -> StepLog:
        """Состояние неожиданно изменилось: строим новый путь к нужному ребру (DP)."""
        if not allow_replan or replans >= self.config.max_replans:
            log.errors.append(EngineError(ErrorCode.STATE_DRIFT, f"{step.device}: перестроение запрещено/исчерпано"))
            return log
        fsm = self.ruleset.fsms[self.world.type_of(step.device)]
        if state == step.target and step.prop is None:
            log.status = StepStatus.ALREADY
            log.trace.append("после синхронизации устройство уже в целевом состоянии")
            return log
        context = self.world.guard_context(step.device, step.obj)
        path = plan_path(
            fsm, state, lambda s: any(e.target == step.target for e in fsm.find(s, step.action)),
            context, edge_filter=lambda e: e.auto,
        )
        if path is None:
            log.errors.append(
                EngineError(ErrorCode.NO_PATH, f"{step.device}: из «{state}» нет пути для {step.action}")
            )
            return log
        log.trace.append(
            "перестроение: " + (" → ".join(e.event for e in path) + " → " if path else "") + step.action
        )
        main = next(e for e in fsm.find(path[-1].target if path else state, step.action) if e.target == step.target)
        new_steps = [
            Step(step.device, e.event, e.source, e.target, e, critical=step.critical, auto=True,
                 timeout_ms=step.timeout_ms, latency_ms=step.latency_ms, group=step.group)
            for e in path
        ]
        new_steps.append(
            Step(step.device, step.action, main.source, main.target, main, step.value, step.prop,
                 step.prev_value, step.obj, step.critical, step.auto, step.reason,
                 step.timeout_ms, step.latency_ms, step.attempts, step.group)
        )
        for sub in new_steps:
            sub_log = self._walk(sub, deadline, max(1, attempts), journal, allow_replan, replans + 1)
            log.trace += [f"  {line}" for line in sub_log.trace]
            log.errors += sub_log.errors
            log.attempts += sub_log.attempts
            if sub_log.status not in (StepStatus.DONE, StepStatus.ALREADY):
                return log
        log.status = StepStatus.DONE
        return log
