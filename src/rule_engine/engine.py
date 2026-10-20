"""Движок: конвейер обработки команды как конечный автомат.

    RECEIVED ─normalize→ NORMALIZED ─parse→ PARSED ─resolve→ RESOLVED
    RESOLVED ─clarify→ AWAITING_CLARIFICATION ─answer→ RESOLVED | ─cancel→ CANCELLED
    RESOLVED ─reject→ REJECTED
    RESOLVED ─plan→ PLANNED ─confirm→ AWAITING_CONFIRMATION ─approve→ PLANNED | ─cancel→ CANCELLED
    PLANNED ─execute→ EXECUTING ─succeed→ SUCCEEDED | ─partial→ PARTIALLY_SUCCEEDED
                                 ─fail→ ROLLING_BACK ─rolled_back→ ROLLED_BACK | ─rollback_failed→ ROLLBACK_FAILED
    PLANNED ─noop→ SUCCEEDED,  PLANNED ─reject→ REJECTED
    RECEIVED ─undo→ ROLLING_BACK  (команда «отмени»)

Запрос не может перепрыгнуть этап: каждый переход проверяется автоматом, а
история переходов попадает в ответ — это и есть трассировка обработки.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rule_engine.errors import EngineError, ErrorCode
from rule_engine.executor import (
    ExecStatus,
    ExecutionConfig,
    ExecutionReport,
    Executor,
    ReliabilityModel,
    SimulatedDriver,
    StepStatus,
    VirtualClock,
)
from rule_engine.fsm import Edge, FSMDefinition, FSMInstance
from rule_engine.index.lexicon import (
    ALL_MATCH_TYPES,
    AddReport,
    Kind,
    MatchType,
    kind_of,
)
from rule_engine.parser import Parse, Parser
from rule_engine.planner import Plan, Planner
from rule_engine.policy import Decision, Policy, PolicyConfig, Status
from rule_engine.resolver import Command, Interpretation, Resolver
from rule_engine.rules import RuleCompiler, RuleSet, apply_overlay_item, scenario_rule
from rule_engine.spec import SceneObject, ScenarioSpec, Spec, parse_scenario, risk_rank, scenario_to_json
from rule_engine.stream import Event, ScenarioTracker
from rule_engine.text.distance import weighted_distance
from rule_engine.text.normalize import NormalizedQuery, Normalizer, basic_clean
from rule_engine.text.stem import make_stemmer
from rule_engine.world import World

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_SPEC = DATA_DIR / "spec.json"

# Состояния запроса
RECEIVED = "RECEIVED"
NORMALIZED = "NORMALIZED"
PARSED = "PARSED"
RESOLVED = "RESOLVED"
AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
PLANNED = "PLANNED"
EXECUTING = "EXECUTING"
ROLLING_BACK = "ROLLING_BACK"
SUCCEEDED = "SUCCEEDED"
PARTIAL = "PARTIALLY_SUCCEEDED"
ROLLED_BACK = "ROLLED_BACK"
ROLLBACK_FAILED = "ROLLBACK_FAILED"
REJECTED = "REJECTED"
CANCELLED = "CANCELLED"
TERMINAL = (SUCCEEDED, PARTIAL, ROLLED_BACK, ROLLBACK_FAILED, REJECTED, CANCELLED)

REQUEST_FSM = FSMDefinition(
    "request",
    [RECEIVED, NORMALIZED, PARSED, RESOLVED, AWAITING_CLARIFICATION, AWAITING_CONFIRMATION,
     PLANNED, EXECUTING, ROLLING_BACK, *TERMINAL],
    RECEIVED,
    [
        Edge("normalize", RECEIVED, NORMALIZED),
        Edge("reject", RECEIVED, REJECTED),
        Edge("undo", RECEIVED, ROLLING_BACK),
        Edge("parse", NORMALIZED, PARSED),
        Edge("reject", NORMALIZED, REJECTED),
        Edge("resolve", PARSED, RESOLVED),
        Edge("clarify", RESOLVED, AWAITING_CLARIFICATION),
        Edge("reject", RESOLVED, REJECTED),
        Edge("plan", RESOLVED, PLANNED),
        Edge("answer", AWAITING_CLARIFICATION, RESOLVED),
        Edge("cancel", AWAITING_CLARIFICATION, CANCELLED),
        Edge("confirm", PLANNED, AWAITING_CONFIRMATION),
        Edge("approve", AWAITING_CONFIRMATION, PLANNED),
        Edge("cancel", AWAITING_CONFIRMATION, CANCELLED),
        Edge("reject", PLANNED, REJECTED),
        Edge("noop", PLANNED, SUCCEEDED),
        Edge("execute", PLANNED, EXECUTING),
        Edge("succeed", EXECUTING, SUCCEEDED),
        Edge("partial", EXECUTING, PARTIAL),
        Edge("fail", EXECUTING, ROLLING_BACK),
        Edge("rolled_back", ROLLING_BACK, ROLLED_BACK),
        Edge("rollback_failed", ROLLING_BACK, ROLLBACK_FAILED),
    ],
    terminal=TERMINAL,
)

YES = frozenset({"да", "ага", "угу", "подтверждаю", "верно", "конечно", "выполняй", "давай", "yes", "ок", "окей"})
NO = frozenset({"нет", "отмена", "отмени", "отменить", "не", "неа", "no", "стоп"})
UNDO = frozenset({"отмени", "отменить", "отмена", "верни", "вернуть", "откати", "undo"})
UNDO_FILLER = frozenset({"как", "было", "назад", "последнее", "последнюю", "команду", "действие", "это", "пожалуйста", "все"})
ALL_WORDS = frozenset({"все", "везде", "оба", "обе", "всех", "всё"})
ORDINALS = {
    "первый": 0, "первое": 0, "первая": 0, "первую": 0, "второй": 1, "второе": 1, "вторая": 1,
    "вторую": 1, "третий": 2, "третье": 2, "третья": 2, "третью": 2, "четвертый": 3,
    "четвертое": 3, "четвертая": 3, "четвертую": 3,
}


@dataclass
class Request:
    id: int
    text: str
    fsm: FSMInstance
    context_room: str | None = None
    query: NormalizedQuery | None = None
    parses: list[Parse] = field(default_factory=list)
    interpretations: list[Interpretation] = field(default_factory=list)
    decision: Decision | None = None
    chosen: Interpretation | None = None
    plan: Plan | None = None
    report: ExecutionReport | None = None
    confirmed: bool = False
    clarify_rounds: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[EngineError] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    offer: str | None = None

    @property
    def state(self) -> str:
        return self.fsm.state

    def fire(self, event: str, note: str = "") -> None:
        self.fsm.fire(event, note=note)


@dataclass
class Session:
    context_room: str | None = None
    pending: Request | None = None
    history: list[Request] = field(default_factory=list)
    offer: str | None = None  # сценарий, предложенный трекером («выполнить остальное?»)


@dataclass
class Response:
    request_id: int
    state: str
    message: str
    decision: str | None = None
    question: str = ""
    options: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    noops: list[str] = field(default_factory=list)
    execution: list[dict[str, Any]] = field(default_factory=list)
    rollback: list[dict[str, Any]] = field(default_factory=list)
    explain: list[str] = field(default_factory=list)
    fired: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "", [], {})}

    def render(self, verbose: bool = False) -> str:
        lines = [f"[{self.state}] {self.message}"]
        if self.question and self.question not in self.message:
            lines.append(self.question)
        lines += [f"  {o}" for o in self.options]
        if verbose:
            if self.plan:
                lines.append("план:")
                lines += [f"  {p}" for p in self.plan]
            if self.noops:
                lines.append("без изменений:")
                lines += [f"  {n}" for n in self.noops]
            for log in self.execution + self.rollback:
                lines.append(f"  · {log['step']}: {log['status']} (попыток {log['attempts']}, {log['elapsed_ms']:.0f} мс)")
                lines += [f"      {t}" for t in log["trace"]]
            if self.errors:
                lines.append("ошибки:")
                lines += [f"  {e['code']} [{e['strategy']}]: {e['message']}" for e in self.errors]
            if self.explain:
                lines.append("объяснение:")
                lines += [f"  {e}" for e in self.explain]
            if self.fired:
                lines.append("правила политики: " + ", ".join(self.fired))
            lines.append("переходы: " + " → ".join(self.transitions))
            lines.append("время, мс: " + ", ".join(f"{k}={v:.3f}" for k, v in self.timings_ms.items()))
        lines += [f"💡 {s}" for s in self.suggestions]
        return "\n".join(lines)


@dataclass
class Understanding:
    """Результат «понимания» без побочных эффектов — для оценки и отладки."""

    query: NormalizedQuery
    parses: list[Parse]
    interpretations: list[Interpretation]
    decision: Decision
    timings: dict[str, float]


class Engine:
    def __init__(
        self,
        ruleset: RuleSet,
        *,
        seed: int = 0,
        policy: PolicyConfig | None = None,
        execution: ExecutionConfig | None = None,
        overlay_path: str | Path | None = None,
        beam: int | None = None,
    ) -> None:
        self.ruleset = ruleset
        self.spec = ruleset.spec
        self.lexicon = ruleset.lexicon
        self.clock = VirtualClock()
        self.world = World(ruleset, self.clock)
        self.driver = SimulatedDriver(ruleset, self.world, self.clock, seed)
        self.reliability = ReliabilityModel(ruleset)
        self.executor = Executor(
            ruleset, self.world, self.driver, self.clock, self.reliability,
            execution or ExecutionConfig.from_dict(self.spec.execution), seed,
        )
        self.normalizer = Normalizer(self.lexicon.stemmer, known_word=self.lexicon.known_word)
        policy_config = policy or PolicyConfig.from_dict(self.spec.policy)
        self.parser = Parser(self.lexicon, beam or int(self.spec.policy.get("beam", 8)))
        self.resolver = Resolver(ruleset)
        self.policy = Policy(ruleset, policy_config)
        self.planner = Planner(ruleset)
        self.tracker = ScenarioTracker(ruleset)
        self.session = Session()
        self.overlay_path = Path(overlay_path) if overlay_path else None
        self.last_success: Request | None = None
        self._narrowed: list[Interpretation] | None = None
        self.overlay_scenarios: list[dict[str, Any]] = []
        self._offer: str | None = None
        self.event_log: list[dict[str, Any]] = []
        self._next_id = 1

    # ----------------------------------------------------------- construction
    @classmethod
    def from_files(
        cls,
        spec_path: str | Path | None = None,
        overlay_path: str | Path | None = None,
        *,
        backend: str = "trie",
        stemmer: str = "light",
        match_types: frozenset[MatchType] = ALL_MATCH_TYPES,
        **kwargs: Any,
    ) -> Engine:
        spec = Spec.load(spec_path or DEFAULT_SPEC)
        overlay = load_overlay(overlay_path)
        for raw in overlay["scenarios"]:
            spec.add_scenario(parse_scenario(raw))
        ruleset = RuleCompiler(make_stemmer(stemmer), backend, match_types).compile(spec, overlay["aliases"])
        engine = cls(ruleset, overlay_path=overlay_path, **kwargs)
        engine.overlay_scenarios = list(overlay["scenarios"])
        return engine

    # ---------------------------------------------------------- understanding
    def understand(self, text: str, context_room: str | None = None) -> Understanding:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        query = self.normalizer.normalize(text)
        t1 = time.perf_counter()
        parses = self.parser.parse(query)
        t2 = time.perf_counter()
        interps = self.resolver.resolve(parses, query, self.world, context_room)
        t3 = time.perf_counter()
        decision = self.policy.decide(query, interps)
        t4 = time.perf_counter()
        timings.update(
            normalize=(t1 - t0) * 1000, parse=(t2 - t1) * 1000, resolve=(t3 - t2) * 1000, decide=(t4 - t3) * 1000
        )
        return Understanding(query, parses, interps, decision, timings)

    # ---------------------------------------------------------------- handle
    def handle(
        self, text: str, session: Session | None = None, context_room: str | None = None
    ) -> Response:
        session = session or self.session
        offer, session.offer = session.offer, None
        self._offer = None
        if offer is not None and session.pending is None and _is_yes(text):
            response = self._accept_offer(offer, text, session)
        else:
            response = self._handle(text, session, context_room)
        session.offer = self._offer
        return response

    def _accept_offer(self, scenario_id: str, text: str, session: Session) -> Response:
        """«да» в ответ на подсказку трекера: выполнить сценарий целиком.

        Уже сделанные шаги станут идемпотентными no-op, опасные рёбра всё равно
        потребуют подтверждения на уровне плана.
        """
        request = self._new_request(text, session.context_room)
        session.history.append(request)
        for event in ("normalize", "parse", "resolve"):
            request.fire(event)
        scenario = self.spec.scenarios[scenario_id]
        interp = Interpretation(
            (Command(action=None, scenario=scenario_id, notes=("принята подсказка трекера",)),),
            1.0, [{}], [f"подсказка трекера → сценарий «{scenario.title}»"],
        )
        return self._after_decision(request, Decision(Status.OK, chosen=interp, fired=["offer-accepted"]), session)

    def _handle(self, text: str, session: Session, context_room: str | None) -> Response:
        if session.pending is not None:
            return self._answer(session, text, context_room)
        request = self._new_request(text, context_room or session.context_room)
        session.history.append(request)
        if self._is_undo(text):
            return self._undo(request)
        understanding = self.understand(text, request.context_room)
        request.query = understanding.query
        request.parses = understanding.parses
        request.interpretations = understanding.interpretations
        request.decision = understanding.decision
        request.timings.update(understanding.timings)
        if not understanding.query.tokens:
            request.fire("reject")
            request.errors.append(EngineError(ErrorCode.EMPTY_INPUT))
            return self._respond(request)
        request.fire("normalize")
        request.fire("parse")
        request.fire("resolve")
        return self._after_decision(request, understanding.decision, session)

    def _new_request(self, text: str, context_room: str | None) -> Request:
        request = Request(self._next_id, text, FSMInstance(REQUEST_FSM, clock=self.clock), context_room)
        self._next_id += 1
        return request

    def _after_decision(self, request: Request, decision: Decision, session: Session) -> Response:
        request.decision = decision
        request.errors = list(decision.errors)
        if decision.status in (Status.REJECT, Status.NOT_FOUND):
            request.fire("reject")
            return self._respond(request)
        if decision.status is Status.CLARIFY:
            request.fire("clarify")
            session.pending = request
            return self._respond(request)
        request.chosen = decision.chosen or decision.top
        assert request.chosen is not None
        t0 = time.perf_counter()
        plan = self.planner.plan(request.chosen, self.world)
        request.timings["plan"] = (time.perf_counter() - t0) * 1000
        request.plan = plan
        request.fire("plan")
        if plan.errors:
            request.errors = plan.errors
            request.fire("reject")
            return self._respond(request)
        needs_confirm = decision.status is Status.CONFIRM or (
            plan.risk == "high" and self.policy.config.confirm_high_risk
        )
        if needs_confirm and not request.confirmed:
            if decision.status is not Status.CONFIRM:
                request.errors.append(EngineError(ErrorCode.HIGH_RISK, "план содержит опасный переход"))
            request.fire("confirm")
            session.pending = request
            return self._respond(request)
        return self._execute(request)

    def _execute(self, request: Request) -> Response:
        plan = request.plan
        assert plan is not None
        if not plan.steps:
            request.fire("noop")
            return self._respond(request)
        request.fire("execute")
        t0 = time.perf_counter()
        report = self.executor.execute(plan)
        request.timings["execute_cpu"] = (time.perf_counter() - t0) * 1000
        request.timings["execute_virtual"] = report.elapsed_ms
        request.report = report
        request.errors = list(report.errors)
        if report.status is ExecStatus.SUCCEEDED:
            request.fire("succeed")
        elif report.status is ExecStatus.PARTIAL:
            request.fire("partial")
        else:
            request.fire("fail")
            request.fire("rolled_back" if report.status is ExecStatus.ROLLED_BACK else "rollback_failed")
        if report.status in (ExecStatus.SUCCEEDED, ExecStatus.PARTIAL):
            self.last_success = request
            self._observe(request)
            self._offer = request.offer
        return self._respond(request)

    def _observe(self, request: Request) -> None:
        """Передать выполненные команды в трекер сценариев и журнал для майнинга."""
        assert request.report is not None and request.chosen is not None
        scenario_run = any(c.scenario for c in request.chosen.commands)
        if scenario_run:
            for c in request.chosen.commands:
                if c.scenario:
                    self.tracker.reset(c.scenario)
        events = []
        for log in request.report.done_steps():
            step = log.step
            if step.auto:
                continue
            event = Event(self.clock(), step.action, step.device, self.world.type_of(step.device),
                          self.world.room_of(step.device))
            events.append(event)
            self.event_log.append(
                {"ts": event.ts, "action": step.action, "device": step.device, "request": request.id}
            )
        if not scenario_run:
            for suggestion in self.tracker.observe_batch(events):
                request.suggestions.append(suggestion.message(self.spec))
                if not suggestion.completed and request.offer is None:
                    request.offer = suggestion.scenario

    # ---------------------------------------------------------------- dialog
    def _answer(self, session: Session, text: str, context_room: str | None) -> Response:
        request = session.pending
        assert request is not None
        words = basic_clean(text).split()
        if request.state == AWAITING_CONFIRMATION:
            if _is_yes(text):
                session.pending = None
                request.confirmed = True
                request.fire("approve", note=text)
                return self._execute(request)
            session.pending = None
            request.fire("cancel", note=text)
            if words and all(w in NO for w in words):
                return self._respond(request, "Отменено.")
            return self.handle(text, session, context_room)

        # AWAITING_CLARIFICATION
        if words and all(w in NO for w in words):
            session.pending = None
            request.fire("cancel", note=text)
            return self._respond(request, "Хорошо, ничего не делаю.")
        assert request.decision is not None
        options = request.decision.options
        chosen = self._pick_option(options, words, text, request)
        request.clarify_rounds += 1
        if chosen is None:
            if request.clarify_rounds >= 3:
                session.pending = None
                request.fire("cancel", note=text)
                return self._respond(request, "Не удалось уточнить команду — отменяю.")
            narrowed, self._narrowed = self._narrowed, None
            if narrowed:
                request.decision = Decision(Status.CLARIFY, options=narrowed, errors=request.decision.errors,
                                            fired=request.decision.fired + ["dialog-narrowed"])
                return self._respond(request)
            # ответ не похож на уточнение — считаем его новой командой
            session.pending = None
            request.fire("cancel", note=text)
            return self.handle(text, session, context_room)
        session.pending = None
        request.fire("answer", note=text)
        decision = self._recheck(chosen, request)
        return self._after_decision(request, decision, session)

    def _pick_option(
        self, options: list[Interpretation], words: list[str], text: str, request: Request
    ) -> Interpretation | None:
        self._narrowed = None
        missing_value = any(i.code is ErrorCode.MISSING_VALUE for o in options for i in o.issues)
        if missing_value:
            numbers = [t.value for t in self.normalizer.normalize(text).tokens if t.value is not None]
            if numbers:
                return _with_value(options[0], numbers[0], self.resolver, self.ruleset)
        if len(words) == 1:
            word = words[0]
            index = ORDINALS.get(word)
            if index is None and word.isdigit():
                index = int(word) - 1
            if index is not None:
                return options[index] if 0 <= index < len(options) else None
        if words and all(w in ALL_WORDS for w in words):
            merged = _merge_options(options)
            if merged is not None:
                return merged
        query = self.normalizer.normalize(text)
        refs = {s.entry.ref for s in self.lexicon.match_spans(query.tokens) if s.strict or s.quality >= 0.7}
        if not refs:
            return None
        matching = [o for o in options if all(_consistent(o, ref, self) for ref in refs)]
        if len(matching) == 1:
            return matching[0]
        if 1 < len(matching) < len(options):
            self._narrowed = matching
        return None

    def _recheck(self, chosen: Interpretation, request: Request) -> Decision:
        """После уточнения: остались ли проблемы и нужна ли проверка риска."""
        remaining = [
            i for i in chosen.issues
            if i.code not in (ErrorCode.MISSING_TARGET, ErrorCode.MISSING_ACTION, ErrorCode.AMBIGUOUS)
        ]
        if any(i.code is ErrorCode.MISSING_VALUE for i in remaining):
            return Decision(Status.CLARIFY, options=[chosen], errors=remaining, fired=["dialog-missing-value"])
        hard = [i for i in remaining if i.strategy.value == "reject"]
        if hard:
            return Decision(Status.REJECT, options=[chosen], errors=hard, fired=["dialog-hard-issue"])
        chosen = _refresh_risk(chosen, self.ruleset)
        assert request.query is not None
        risk_decision = self.policy._risk(request.query, [chosen])
        if risk_decision is not None:
            risk_decision.fired.append("dialog-risk")
            return risk_decision
        return Decision(Status.OK, chosen=chosen, fired=["dialog-ok"])

    # ------------------------------------------------------------------ undo
    def _is_undo(self, text: str) -> bool:
        words = basic_clean(text).split()
        return bool(words) and any(w in UNDO for w in words) and all(w in UNDO | UNDO_FILLER for w in words)

    def _undo(self, request: Request) -> Response:
        last = self.last_success
        if last is None or last.report is None or not last.report.journal:
            request.fire("reject")
            return self._respond(request, "Нечего отменять.")
        request.fire("undo")
        rollback = self.executor.compensate(last.report.journal)
        request.report = ExecutionReport(ExecStatus.ROLLED_BACK, rollback=rollback)
        failed = [r for r in rollback if r.status is StepStatus.COMPENSATION_FAILED]
        for r in failed:
            request.errors += r.errors
        request.fire("rollback_failed" if failed else "rolled_back")
        self.last_success = None
        return self._respond(request, "Не удалось полностью отменить." if failed else f"Отменил: «{last.text}».")

    # ------------------------------------------------------------ vocabulary
    def add_alias(self, ref: str, alias: str, hint: str | None = None) -> AddReport:
        """Добавить синоним без пересборки индекса и без порчи старых записей."""
        self._check_ref(ref)
        version = self.lexicon.version + 1
        item = {"ref": ref, "alias": alias, "version": version}
        if hint:
            item["hint"] = hint
        before = len(self.lexicon.entries)
        apply_overlay_item(self.ruleset, item)
        if len(self.lexicon.entries) == before:
            self.ruleset.overlay.pop()
            return AddReport(entry=None, version=self.lexicon.version, duplicate=True)
        entry = self.lexicon.entries[-1]
        conflicts = [e for e in self.lexicon.refs_for_alias(alias) if e.ref != ref and e.kind is entry.kind]
        self._save_overlay()
        return AddReport(entry=entry, version=self.lexicon.version, conflicts=conflicts)

    def alias_warnings(self, ref: str, alias: str) -> list[str]:
        """Проверка нового синонима действия на опасную близость к антониму."""
        if kind_of(ref) is not Kind.ACTION:
            return []
        antonym = self.ruleset.antonym(ref.split(":", 1)[1])
        if not antonym:
            return []
        warnings = []
        for word in basic_clean(alias).split():
            for other in self.ruleset.aliases_of(f"action:{antonym}"):
                for other_word in other.split():
                    d = weighted_distance(word, other_word, 1.6)
                    if d <= 1.6:
                        warnings.append(f"«{word}» близко к «{other_word}» ({antonym}), d={d:.1f}")
        return warnings

    def rollback_aliases(self, version: int) -> int:
        removed = self.lexicon.rollback(version)
        self.ruleset.overlay = [i for i in self.ruleset.overlay if int(i["version"]) <= version]
        self._save_overlay()
        return removed

    def _check_ref(self, ref: str) -> None:
        kind = kind_of(ref)
        ident = ref.split(":", 1)[1]
        tables = {
            Kind.ACTION: self.spec.actions, Kind.DEVICE: self.spec.devices,
            Kind.DEVICE_TYPE: self.spec.device_types, Kind.ROOM: self.spec.rooms,
            Kind.SCENARIO: self.spec.scenarios, Kind.OBJECT_TYPE: self.spec.object_types,
        }
        if kind is Kind.ATTRIBUTE:
            name, _, value = ident.partition("=")
            if value not in self.spec.attributes.get(name, {}):
                raise KeyError(ref)
        elif ident not in tables[kind]:
            raise KeyError(ref)

    def _save_overlay(self) -> None:
        if self.overlay_path is None:
            return
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"aliases": self.ruleset.overlay, "scenarios": self.overlay_scenarios}
        self.overlay_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ misc
    def update_scene(self, objects: list[dict[str, Any]] | list[SceneObject]) -> None:
        parsed = [o if isinstance(o, SceneObject) else SceneObject.parse(o) for o in objects]
        self.world.update_scene(parsed)

    def add_scenario(self, scenario: ScenarioSpec, persist: bool = True) -> None:
        """Добавить сценарий (например, выученный) без пересборки движка."""
        self.spec.add_scenario(scenario)
        for alias in scenario.aliases:
            self.lexicon.add(alias.alias, f"scenario:{scenario.id}", source=scenario.source)
        self.ruleset.rules.append(scenario_rule(scenario))
        self.tracker.refresh()
        if persist:
            self.overlay_scenarios = [s for s in self.overlay_scenarios if s["id"] != scenario.id]
            self.overlay_scenarios.append(scenario_to_json(scenario))
            self._save_overlay()

    def describe_world(self) -> list[str]:
        lines = []
        for device_id in sorted(self.world.devices):
            spec = self.spec.devices[device_id]
            lines.append(f"{device_id:<20} {spec.room or '-':<10} {self.world.describe(device_id)}")
        if self.world.scene:
            lines.append("сцена: " + ", ".join(
                f"{o.id}({o.type}, {', '.join(f'{k}={v}' for k, v in o.attrs.items())}, {o.confidence:.2f})"
                for o in self.world.scene.values()
            ))
        return lines

    # -------------------------------------------------------------- response
    def render_interpretation(self, interp: Interpretation) -> str:
        return "; ".join(self.render_command(c) for c in interp.commands)

    def render_command(self, command: Command) -> str:
        if command.scenario:
            return f"сценарий «{self.spec.scenarios[command.scenario].title}»"
        action = self.spec.actions[command.action].title if command.action else "?"
        targets = ", ".join(self.spec.devices[d].name for d in command.targets)
        if command.objects:
            objs = []
            for o in command.objects:
                obj = self.world.scene.get(o)
                objs.append(f"{o} ({', '.join(obj.attrs.values())})" if obj else o)
            targets = f"{', '.join(objs)} [{targets}]"
        value = f" = {command.value:g}" if command.value is not None else ""
        return f"{action}: {targets}{value}"

    def _respond(self, request: Request, message: str | None = None) -> Response:
        decision = request.decision
        state = request.state
        options: list[str] = []
        question = ""
        if state == AWAITING_CLARIFICATION and decision is not None:
            question = self._question(request, decision)
            limit = self.policy.config.max_options
            options = [f"{i + 1}) {self.render_interpretation(o)}" for i, o in enumerate(decision.options[:limit])]
            if len(decision.options) > limit:
                options.append(f"… и ещё {len(decision.options) - limit} (можно ответить «везде»)")
            if any(e.code is ErrorCode.MISSING_VALUE for e in request.errors):
                options = []
        elif state == AWAITING_CONFIRMATION and request.plan is not None and request.chosen is not None:
            reasons = "; ".join(e.message() for e in request.errors) or "опасная команда"
            question = (
                f"{reasons} Выполнить «{self.render_interpretation(request.chosen)}»? "
                f"(шаги: {', '.join(s.describe() for s in request.plan.steps)}) — да/нет"
            )
        if message is None:
            message = self._message(request, question)
        report = request.report
        response = Response(
            request_id=request.id,
            state=state,
            message=message,
            decision=decision.status.value if decision else None,
            question=question,
            options=options,
            errors=[e.to_dict() for e in request.errors],
            plan=[s.describe() for s in request.plan.steps] if request.plan else [],
            noops=list(request.plan.noops + request.plan.skipped) if request.plan else [],
            execution=[_log_dict(log) for log in report.logs] if report else [],
            rollback=[_log_dict(log) for log in report.rollback] if report else [],
            explain=(
                (request.query.trace if request.query else [])
                + ((request.chosen or (decision.top if decision else None)).explain
                   if (request.chosen or (decision and decision.top)) else [])
            ),
            fired=decision.fired if decision else [],
            suggestions=request.suggestions,
            transitions=[f"{h.source}→{h.target}" for h in request.fsm.history],
            timings_ms={k: round(v, 4) for k, v in request.timings.items()},
        )
        return response

    def _question(self, request: Request, decision: Decision) -> str:
        """Вопрос по существу различий между вариантами."""
        errors = request.errors
        codes = {e.code for e in errors}
        options = decision.options
        if ErrorCode.MISSING_VALUE in codes and options:
            command = next((c for c in options[0].commands if c.action), None)
            if command is not None:
                bounds = [
                    e.meta.get("range")
                    for d in command.targets
                    for e in self.ruleset.fsms[self.spec.devices[d].type].edges_for(command.action)
                    if e.meta.get("range")
                ]
                hint = f" (от {bounds[0][0]:g} до {bounds[0][1]:g})" if bounds else ""
                return f"Какое значение: {self.spec.actions[command.action].title}{hint}?"
        singles = [o.commands[0] for o in options if len(o.commands) == 1]
        if len(singles) == len(options) and len(options) > 1:
            actions = list(dict.fromkeys(c.action for c in singles))
            target_sets = {c.targets for c in singles}
            if len(actions) > 1 and len(target_sets) == 1 and None not in actions:
                titles = " или ".join(self.spec.actions[a].title for a in actions)
                return f"{titles.capitalize()}?"
            if len(actions) == 1 and actions[0] is not None:
                rooms = [{self.spec.devices[d].room for d in c.targets} for c in singles]
                if all(len(r) == 1 for r in rooms) and len({next(iter(r)) for r in rooms}) == len(rooms):
                    names = [self.spec.rooms[next(iter(r))].aliases[0].alias for r in rooms if next(iter(r))]
                    return f"В какой комнате {self.spec.actions[actions[0]].title}: {', '.join(names)}?"
                if any(c.objects for c in singles):
                    return f"Какой объект {self.spec.actions[actions[0]].title}?"
                return f"Какое устройство {self.spec.actions[actions[0]].title}?"
        return errors[0].message() if errors else "Уточните, пожалуйста."

    def _message(self, request: Request, question: str) -> str:
        state = request.state
        chosen = request.chosen
        if state == SUCCEEDED:
            if request.plan and not request.plan.steps:
                return "Уже сделано: " + "; ".join(request.plan.noops)
            return "Готово: " + (self.render_interpretation(chosen) if chosen else "")
        if state == PARTIAL:
            return "Сделано частично: " + "; ".join(e.message() for e in request.errors)
        if state == ROLLED_BACK:
            reason = request.errors[0].message() if request.errors else ""
            return f"Не получилось ({reason}). Все изменения откатил."
        if state == ROLLBACK_FAILED:
            return "Ошибка и откат не удался — нужна проверка: " + "; ".join(e.message() for e in request.errors)
        if state == REJECTED:
            return "; ".join(e.message() for e in request.errors) or "Отказ."
        if state == CANCELLED:
            return "Отменено."
        return question


def _is_yes(text: str) -> bool:
    words = basic_clean(text).split()
    return bool(words) and all(w in YES or w == "пожалуйста" for w in words)


def _log_dict(log) -> dict[str, Any]:
    return {
        "step": log.step.describe(),
        "status": log.status.value,
        "attempts": log.attempts,
        "elapsed_ms": round(log.elapsed_ms, 1),
        "errors": [e.code.value for e in log.errors],
        "trace": log.trace,
    }


def _consistent(option: Interpretation, ref: str, engine: Engine) -> bool:
    kind, ident = ref.split(":", 1)
    spec = engine.spec
    for command in option.commands:
        if kind == "action" and command.action == ident:
            return True
        if kind == "scenario" and command.scenario == ident:
            return True
        if kind == "device" and ident in command.targets:
            return True
        if kind == "type" and any(spec.devices[d].type == ident for d in command.targets):
            return True
        if kind == "room" and any(spec.devices[d].room == ident for d in command.targets):
            return True
        if kind == "object" and any(
            engine.world.scene.get(o) and engine.world.scene[o].type == ident for o in command.objects
        ):
            return True
        if kind == "attr":
            name, _, value = ident.partition("=")
            if any(engine.world.scene.get(o) and engine.world.scene[o].attrs.get(name) == value
                   for o in command.objects):
                return True
    return False


def _merge_options(options: list[Interpretation]) -> Interpretation | None:
    """Ответ «все/везде» на вопрос «в какой комнате?»: объединить цели вариантов."""
    if not options or any(len(o.commands) != 1 for o in options):
        return None
    actions = {o.commands[0].action for o in options}
    if len(actions) != 1 or None in actions or any(o.commands[0].objects for o in options):
        return None
    first = options[0]
    targets = tuple(dict.fromkeys(d for o in options for d in o.commands[0].targets))
    command = replace(first.commands[0], targets=targets, issues=())
    return Interpretation((command,), first.score, first.features, first.explain + ["объединено по ответу «все»"])


def _with_value(option: Interpretation, value: float, resolver: Resolver, ruleset: RuleSet) -> Interpretation:
    commands = []
    for command in option.commands:
        if any(i.code is ErrorCode.MISSING_VALUE for i in command.issues) and command.action:
            types = {ruleset.spec.devices[d].type for d in command.targets}
            issues = tuple(i for i in command.issues if i.code is not ErrorCode.MISSING_VALUE)
            issues += tuple(resolver._range_issues(command.action, types, value))
            command = replace(command, value=value, issues=issues)
        commands.append(command)
    return Interpretation(tuple(commands), option.score, option.features, option.explain + [f"значение {value:g} из ответа"])


def _refresh_risk(interp: Interpretation, ruleset: RuleSet) -> Interpretation:
    commands = []
    for command in interp.commands:
        if command.action:
            types = {ruleset.spec.devices[d].type for d in command.targets}
            risks = [ruleset.spec.actions[command.action].risk] + [
                ruleset.max_action_risk(command.action, t) for t in types if t in ruleset.fsms
            ]
            command = replace(command, risk=max(risks, key=risk_rank))
        commands.append(command)
    return Interpretation(tuple(commands), interp.score, interp.features, interp.explain)


def load_overlay(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    """Оверлей — версионированные дополнения к спецификации: синонимы и выученные сценарии."""
    if path is None or not Path(path).exists():
        return {"aliases": [], "scenarios": []}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"aliases": list(data.get("aliases", [])), "scenarios": list(data.get("scenarios", []))}
