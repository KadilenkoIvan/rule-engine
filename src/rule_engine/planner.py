"""Планирование: превращение команд в последовательность рёбер автоматов.

Для каждого устройства-цели:

1. ребро действия есть из текущего состояния → один шаг;
2. устройство уже в целевом состоянии (и значение совпадает) → NO-OP:
   повторное «включи свет» не ошибка, а идемпотентная команда;
3. ребра нет → кратчайший путь по *автоматическим* рёбрам до состояния, где
   ребро доступно (DP Беллмана–Форда, :func:`rule_engine.fsm.plan_path`);
   так «возьми чашку» при спящем манипуляторе становится «включить → взять»;
4. пути нет → PRECONDITION_FAILED с объяснением, из каких состояний можно.

Далее :func:`allocate_attempts` распределяет повторы по шагам под общий
дедлайн — DP «рюкзак с выбором» максимизирует вероятность успеха всей саги.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rule_engine.errors import EngineError, ErrorCode
from rule_engine.fsm import Edge, plan_path
from rule_engine.resolver import Command, Interpretation
from rule_engine.rules import RuleSet
from rule_engine.spec import ScenarioSpec, max_risk
from rule_engine.world import World


@dataclass
class Step:
    device: str
    action: str
    source: str
    target: str
    edge: Edge
    value: float | None = None
    prop: str | None = None
    prev_value: float | None = None
    obj: str | None = None
    critical: bool = True
    auto: bool = False  # вставлен планировщиком
    reason: str = ""
    timeout_ms: float = 2000.0
    latency_ms: float = 150.0
    attempts: int = 1
    group: int = 0  # номер команды/шага сценария, к которому относится шаг

    @property
    def risk(self) -> str:
        return self.edge.risk

    def describe(self) -> str:
        val = f"={self.value:g}" if self.value is not None else ""
        obj = f" [{self.obj}]" if self.obj else ""
        auto = " (авто)" if self.auto else ""
        return f"{self.device}: {self.source} --{self.action}{val}{obj}--> {self.target}{auto}"


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    noops: list[str] = field(default_factory=list)
    errors: list[EngineError] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # некритичные шаги, которые не спланировались

    @property
    def risk(self) -> str:
        return max_risk(*(s.risk for s in self.steps)) if self.steps else "low"

    @property
    def ok(self) -> bool:
        return not self.errors


class Planner:
    def __init__(self, ruleset: RuleSet) -> None:
        self.ruleset = ruleset
        self.spec = ruleset.spec

    def plan(self, interpretation: Interpretation, world: World) -> Plan:
        plan = Plan()
        virtual: dict[str, tuple[str, dict[str, float]]] = {}  # состояние по ходу плана
        for group, command in enumerate(interpretation.commands):
            if command.scenario:
                self._plan_scenario(self.spec.scenarios[command.scenario], world, plan, virtual, group)
            else:
                self._plan_command(command, world, plan, virtual, group, critical=True)
        return plan

    # -------------------------------------------------------------- commands
    def _plan_command(
        self,
        command: Command,
        world: World,
        plan: Plan,
        virtual: dict[str, tuple[str, dict[str, float]]],
        group: int,
        critical: bool,
    ) -> None:
        assert command.action is not None
        obj = command.objects[0] if command.objects else None
        for device in command.targets:
            self._plan_device(command.action, device, command.value, obj, world, plan, virtual, group, critical)

    def _plan_scenario(
        self,
        scenario: ScenarioSpec,
        world: World,
        plan: Plan,
        virtual: dict[str, tuple[str, dict[str, float]]],
        group: int,
    ) -> None:
        for idx, step in enumerate(scenario.steps):
            sel = step.target
            if sel.device:
                devices = [sel.device]
            else:
                devices = [d.id for d in self.spec.devices_of(sel.type or "", sel.room)]
            for device in devices:
                dtype = self.spec.devices[device].type
                if not self.ruleset.applicable(step.action, dtype):
                    plan.skipped.append(f"{scenario.id}#{idx}: {step.action} неприменимо к {device}")
                    continue
                self._plan_device(
                    step.action, device, step.value, None, world, plan, virtual, group * 100 + idx, step.critical
                )

    def _plan_device(
        self,
        action: str,
        device: str,
        value: float | None,
        obj: str | None,
        world: World,
        plan: Plan,
        virtual: dict[str, tuple[str, dict[str, float]]],
        group: int,
        critical: bool,
    ) -> None:
        dev_spec = self.spec.devices[device]
        dtype = self.spec.device_types[dev_spec.type]
        fsm = self.ruleset.fsms[dev_spec.type]
        state, values = virtual.get(device, (world.state(device), dict(world.devices[device].values)))
        name = dev_spec.name

        def fail(code: ErrorCode, detail: str) -> None:
            error = EngineError(code, detail, {"device": device, "action": action})
            if critical:
                plan.errors.append(error)
            else:
                plan.skipped.append(f"{device}: {error.message()}")

        if not world.devices[device].online:
            fail(ErrorCode.DEVICE_OFFLINE, f"{name} не в сети")
            return
        context = world.guard_context(device, obj)
        action_edges = fsm.edges_for(action)
        holding = world.devices[device].holding
        if action == "release" and obj is None:
            obj = holding  # чтобы откат («взять обратно») знал, какой объект
        elif obj is not None and holding is not None and self.spec.actions[action].target_kind == "object":
            if holding == obj:
                plan.noops.append(f"{name}: уже держит {obj}")
            else:
                fail(ErrorCode.PRECONDITION_FAILED, f"{name} уже держит {holding} — сначала положите его")
            return
        prop = next((e.meta.get("value") for e in action_edges if e.meta.get("value")), None)

        # Идемпотентность: уже в целевом состоянии и значение то же.
        already_there = (
            prop is None
            and action_edges
            and all(e.target == state for e in action_edges)
            and not fsm.find(state, action)
        )
        if already_there:
            plan.noops.append(f"{name}: уже «{state}» — {action} не нужен")
            return
        same_value = (
            prop is not None
            and value is not None
            and values.get(prop) == value
            and any(e.target == state for e in fsm.find(state, action))
        )
        if same_value:
            plan.noops.append(f"{name}: {prop} уже {value:g}")
            return

        def enabled(s: str) -> bool:
            return any(e.allowed(context) for e in fsm.find(s, action))

        path = plan_path(fsm, state, enabled, context, edge_filter=lambda e: e.auto)
        if path is None:
            sources = sorted({e.source for e in action_edges})
            failed_guards = sorted({e.guard_name for e in fsm.find(state, action) if not e.allowed(context)})
            if failed_guards:
                fail(ErrorCode.PRECONDITION_FAILED, f"{name}: не выполнено условие {', '.join(failed_guards)}")
            else:
                fail(
                    ErrorCode.PRECONDITION_FAILED,
                    f"{name} в состоянии «{state}»; {action} возможно только из: {', '.join(sources) or '—'}",
                )
            return

        steps: list[Step] = []
        current = state
        for edge in path:
            steps.append(self._step(device, edge, None, None, values, None, critical, True, group, dtype))
            steps[-1].reason = f"нужно для {action}: {edge}"
            current = edge.target
        main = next(e for e in fsm.find(current, action) if e.allowed(context))
        edge_prop = main.meta.get("value")
        steps.append(self._step(device, main, value, edge_prop, values, obj, critical, False, group, dtype))
        plan.steps.extend(steps)
        new_values = dict(values)
        if edge_prop and value is not None:
            new_values[edge_prop] = value
        virtual[device] = (main.target, new_values)

    @staticmethod
    def _step(device, edge, value, prop, values, obj, critical, auto, group, dtype) -> Step:
        return Step(
            device=device,
            action=edge.event,
            source=edge.source,
            target=edge.target,
            edge=edge,
            value=value,
            prop=prop,
            prev_value=values.get(prop) if prop else None,
            obj=obj,
            critical=critical,
            auto=auto,
            timeout_ms=dtype.timeout_ms,
            latency_ms=dtype.latency_ms,
            group=group,
        )


def allocate_attempts(
    steps: list[Step],
    success_prob: list[float],
    deadline_ms: float,
    max_attempts: int = 4,
    backoff_ms: float = 100.0,
    backoff_factor: float = 2.0,
    quantum_ms: float = 50.0,
) -> tuple[list[int], float]:
    """Сколько попыток дать каждому шагу, чтобы уложиться в дедлайн.

    Стоимость k попыток шага i: первая — номинальная задержка, каждая
    следующая — худший случай (таймаут) плюс пауза backoff:

        cost_i(k) = latency_i + Σ_{j=1}^{k-1} (timeout_i + backoff·factor^{j-1})

    Вероятность успеха шага за k попыток: P_i(k) = 1 − (1 − p_i)^k.
    DP по шагам и дискретизированному бюджету времени (рюкзак с выбором):

        best[i][b] = max_k  best[i-1][b − cost_i(k)] + log P_i(k)

    Возвращает (попытки по шагам, вероятность успеха всей саги). Если даже по
    одной попытке не укладываемся, всем шагам даётся одна попытка.
    """
    n = len(steps)
    if n == 0:
        return [], 1.0
    budget = int(deadline_ms // quantum_ms)

    def cost(i: int, k: int) -> int:
        total = steps[i].latency_ms
        for j in range(1, k):
            total += steps[i].timeout_ms + backoff_ms * backoff_factor ** (j - 1)
        return math.ceil(total / quantum_ms)

    def log_p(i: int, k: int) -> float:
        p = min(max(success_prob[i], 1e-6), 1 - 1e-9)
        return math.log(1 - (1 - p) ** k)

    neg = -math.inf
    best = [[neg] * (budget + 1) for _ in range(n + 1)]
    choice = [[0] * (budget + 1) for _ in range(n + 1)]
    best[0] = [0.0] * (budget + 1)
    for i in range(1, n + 1):
        for b in range(budget + 1):
            for k in range(1, max_attempts + 1):
                c = cost(i - 1, k)
                if c > b or best[i - 1][b - c] == neg:
                    continue
                value = best[i - 1][b - c] + log_p(i - 1, k)
                if value > best[i][b]:
                    best[i][b], choice[i][b] = value, k
    if best[n][budget] == neg:
        attempts = [1] * n
        return attempts, math.exp(sum(log_p(i, 1) for i in range(n)))
    attempts = [0] * n
    b = budget
    for i in range(n, 0, -1):
        k = choice[i][b]
        attempts[i - 1] = k
        b -= cost(i - 1, k)
    return attempts, math.exp(best[n][budget])
