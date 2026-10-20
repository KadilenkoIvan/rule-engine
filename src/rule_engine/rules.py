"""Компилятор правил: из «набора» (спецификации) составляет «набор правил».

Правила бывают нескольких видов, у каждого есть идентификатор, условие,
заключение и источник — это и делает решения движка объяснимыми:

* LEXICAL       — «слово ~ синоним → сущность» (из них строится лексикон);
* TRANSITION    — ребро автомата устройства: действие, откуда, куда, риск;
* APPLICABILITY — к каким типам устройств применимо действие;
* SAFETY        — риск, подтверждение, запрет нечёткого совпадения;
* CONFUSABLE    — автоматически найденные опасно похожие синонимы разных
                  действий (``включи`` / ``выключи``, d=1) — считаются DP;
* COMPENSATION  — как откатить ребро (путь назад по автомату, тоже DP);
* SCENARIO      — сценарий: последовательность шагов-саги;
* MINED         — правила, выученные из потока команд (:mod:`rule_engine.mining`).

Компилятор также делает статический анализ: недостижимые состояния,
конфликты синонимов, опасные действия без пути отката.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from typing import Any

from rule_engine.fsm import Edge, FSMDefinition, plan_path
from rule_engine.index.lexicon import ALL_MATCH_TYPES, Lexicon, MatchType
from rule_engine.spec import ScenarioSpec, Spec, max_risk
from rule_engine.text.distance import weighted_distance
from rule_engine.text.normalize import FILLERS
from rule_engine.text.stem import Stemmer, light_stem


class RuleKind(StrEnum):
    LEXICAL = "lexical"
    TRANSITION = "transition"
    APPLICABILITY = "applicability"
    SAFETY = "safety"
    CONFUSABLE = "confusable"
    COMPENSATION = "compensation"
    SCENARIO = "scenario"
    MINED = "mined"


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    kind: RuleKind
    when: str
    then: str
    source: str = "spec"
    data: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.id}] ЕСЛИ {self.when} ТО {self.then}"


GUARDS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "object_visible": lambda ctx: ctx.get("object_confidence", 0.0) >= 0.5,
    "object_graspable": lambda ctx: bool(ctx.get("object_graspable", False)),
}

CONFUSABLE_DISTANCE = 1.6


@dataclass(frozen=True, slots=True)
class ConfusablePair:
    alias_a: str
    ref_a: str
    alias_b: str
    ref_b: str
    distance: float
    antonyms: bool


@dataclass
class RuleSet:
    spec: Spec
    lexicon: Lexicon
    fsms: dict[str, FSMDefinition]
    rules: list[Rule]
    confusables: list[ConfusablePair]
    warnings: list[str]
    overlay: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._types_by_action: dict[str, set[str]] = defaultdict(set)
        for type_id, fsm in self.fsms.items():
            for event in fsm.events():
                self._types_by_action[event].add(type_id)

    def types_for_action(self, action: str) -> set[str]:
        return set(self._types_by_action.get(action, ()))

    def applicable(self, action: str, type_id: str) -> bool:
        return type_id in self._types_by_action.get(action, ())

    def edge_risk(self, edge: Edge) -> str:
        action = self.spec.actions[edge.event]
        return max_risk(action.risk, edge.risk)

    def max_action_risk(self, action: str, type_id: str) -> str:
        edges = self.fsms[type_id].edges_for(action) if type_id in self.fsms else []
        return max_risk(self.spec.actions[action].risk, *(e.risk for e in edges))

    def antonym(self, action: str) -> str | None:
        spec = self.spec.actions.get(action)
        return spec.antonym if spec else None

    def aliases_of(self, ref: str) -> list[str]:
        return [e.alias for e in self.lexicon.entries if e.ref == ref]

    def rules_of(self, kind: RuleKind) -> list[Rule]:
        return [r for r in self.rules if r.kind is kind]

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = defaultdict(int)
        for rule in self.rules:
            counts[rule.kind.value] += 1
        return {
            "rules": dict(sorted(counts.items())),
            "lexicon": self.lexicon.stats(),
            "device_types": len(self.fsms),
            "devices": len(self.spec.devices),
            "scenarios": len(self.spec.scenarios),
            "confusable_pairs": len(self.confusables),
            "warnings": len(self.warnings),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "warnings": self.warnings,
            "rules": [
                {"id": r.id, "kind": r.kind.value, "when": r.when, "then": r.then, "source": r.source}
                for r in self.rules
            ],
        }


class RuleCompiler:
    def __init__(
        self,
        stemmer: Stemmer = light_stem,
        backend: str = "trie",
        match_types: frozenset[MatchType] = ALL_MATCH_TYPES,
    ) -> None:
        self.stemmer = stemmer
        self.backend = backend
        self.match_types = match_types

    def compile(self, spec: Spec, overlay: list[dict[str, Any]] | None = None) -> RuleSet:
        rules: list[Rule] = []
        warnings: list[str] = []
        lexicon = Lexicon(self.stemmer, self.backend, self.match_types)
        self._lexical(spec, lexicon, rules, warnings)
        fsms = self._fsms(spec, rules, warnings)
        self._applicability(spec, fsms, rules)
        confusables = self._confusables(spec, lexicon, rules)
        self._safety(spec, fsms, rules, warnings)
        for scenario in spec.scenarios.values():
            rules.append(scenario_rule(scenario))
        ruleset = RuleSet(spec, lexicon, fsms, rules, confusables, warnings)
        for item in overlay or ():
            apply_overlay_item(ruleset, item)
        return ruleset

    # --------------------------------------------------------------- lexicon
    def _lexical(self, spec: Spec, lexicon: Lexicon, rules: list[Rule], warnings: list[str]) -> None:
        def add(ref: str, aliases, source: str = "spec") -> None:
            for a in aliases:
                report = lexicon.add(a.alias, ref, weight=a.weight, hint=a.hint, source=source)
                if report.entry is None:
                    continue
                for other in report.conflicts:
                    warnings.append(
                        f"синоним «{report.entry.alias}» указывает и на {other.ref}, и на {ref} — "
                        "будет переспрашивать"
                    )
                hint = f" (тип: {a.hint})" if a.hint else ""
                rules.append(
                    Rule(
                        f"lex:{report.entry.id}", RuleKind.LEXICAL,
                        f"во фразе есть «{report.entry.alias}»", f"{ref}{hint}", source,
                        {"entry": report.entry.id},
                    )
                )

        for room in spec.rooms.values():
            add(f"room:{room.id}", room.aliases)
        for action in spec.actions.values():
            add(f"action:{action.id}", action.aliases)
        for dtype in spec.device_types.values():
            add(f"type:{dtype.id}", dtype.aliases)
        for device in spec.devices.values():
            add(f"device:{device.id}", device.aliases)
        for obj in spec.object_types.values():
            add(f"object:{obj.id}", obj.aliases)
        for name, values in spec.attributes.items():
            for value, aliases in values.items():
                add(f"attr:{name}={value}", aliases)
        for scenario in spec.scenarios.values():
            add(f"scenario:{scenario.id}", scenario.aliases, scenario.source)

    # ------------------------------------------------------------------ fsm
    def _fsms(self, spec: Spec, rules: list[Rule], warnings: list[str]) -> dict[str, FSMDefinition]:
        fsms = {}
        for dtype in spec.device_types.values():
            edges = []
            for tr in dtype.transitions:
                action = spec.actions[tr.action]
                risk = max_risk(action.risk, tr.risk or action.risk)
                for source in tr.sources:
                    target = source if tr.target == "=" else tr.target
                    edges.append(
                        Edge(
                            event=tr.action, source=source, target=target, cost=tr.cost,
                            risk=risk, auto=tr.auto,
                            guard=GUARDS.get(tr.guard) if tr.guard else None,
                            guard_name=tr.guard or "",
                            meta={
                                "value": tr.value,
                                "range": tr.range,
                                "irreversible": tr.irreversible,
                            },
                        )
                    )
                    rules.append(
                        Rule(
                            f"tr:{dtype.id}:{tr.action}:{source}", RuleKind.TRANSITION,
                            f"устройство типа {dtype.id} в состоянии «{source}» и команда {tr.action}"
                            + (f" и {tr.guard}" if tr.guard else ""),
                            f"перейти в «{target}» (риск {risk}"
                            + (", можно вставлять автоматически" if tr.auto else "") + ")",
                        )
                    )
                if tr.guard and tr.guard not in GUARDS:
                    warnings.append(f"type {dtype.id}: неизвестный guard {tr.guard!r}")
            fsm = FSMDefinition(dtype.id, dtype.states, dtype.initial, edges)
            for state in sorted(fsm.unreachable()):
                warnings.append(f"type {dtype.id}: состояние «{state}» недостижимо из «{dtype.initial}»")
            fsms[dtype.id] = fsm
        return fsms

    def _applicability(self, spec: Spec, fsms: dict[str, FSMDefinition], rules: list[Rule]) -> None:
        for action in spec.actions.values():
            types = sorted(t for t, fsm in fsms.items() if action.id in fsm.events())
            rules.append(
                Rule(
                    f"app:{action.id}", RuleKind.APPLICABILITY,
                    f"команда {action.id}",
                    "применима к типам: " + (", ".join(types) if types else "— (нет устройств)"),
                    data={"types": types},
                )
            )

    # -------------------------------------------------------------- safety
    def _confusables(self, spec: Spec, lexicon: Lexicon, rules: list[Rule]) -> list[ConfusablePair]:
        """Пары синонимов разных действий, отличающиеся одной-двумя правками.

        Попарное взвешенное расстояние (DP) по однословным синонимам действий.
        Такие пары — главный источник «молчаливой» опасной ошибки, поэтому для
        них автоматически создаются правила «при нечётком совпадении переспросить».
        """
        action_words: dict[str, set[str]] = defaultdict(set)
        for entry in lexicon.entries:
            if entry.ref.startswith("action:"):
                for word in entry.tokens:
                    action_words[entry.ref].add(word)
        pairs = []
        for (ref_a, words_a), (ref_b, words_b) in combinations(sorted(action_words.items()), 2):
            a_id, b_id = ref_a.split(":", 1)[1], ref_b.split(":", 1)[1]
            antonyms = spec.actions[a_id].antonym == b_id
            # служебные и короткие слова нечётко не сопоставляются — их пары не опасны
            content_a = {w for w in words_a - words_b if len(w) >= 3 and w not in FILLERS}
            content_b = {w for w in words_b - words_a if len(w) >= 3 and w not in FILLERS}
            for wa in sorted(content_a):
                for wb in sorted(content_b):
                    d = weighted_distance(wa, wb, CONFUSABLE_DISTANCE)
                    if d <= CONFUSABLE_DISTANCE:
                        pairs.append(ConfusablePair(wa, ref_a, wb, ref_b, d, antonyms))
                        rules.append(
                            Rule(
                                f"conf:{a_id}:{wa}~{b_id}:{wb}", RuleKind.CONFUSABLE,
                                f"слово распознано нечётко и близко и к «{wa}», и к «{wb}» (d={d:.1f})",
                                "переспросить" + (" (антонимы!)" if antonyms else ""),
                                source="auto",
                                data={"distance": d, "antonyms": antonyms},
                            )
                        )
        return pairs

    def _safety(
        self, spec: Spec, fsms: dict[str, FSMDefinition], rules: list[Rule], warnings: list[str]
    ) -> None:
        for type_id, fsm in fsms.items():
            for edge in fsm.edges:
                if edge.risk == "high":
                    rules.append(
                        Rule(
                            f"safe:{type_id}:{edge.event}:{edge.source}", RuleKind.SAFETY,
                            f"план содержит ребро {type_id}: {edge}",
                            "требовать подтверждение; нечёткое совпадение глагола запрещено",
                        )
                    )
                if edge.meta.get("irreversible"):
                    rules.append(
                        Rule(
                            f"comp:{type_id}:{edge.event}:{edge.source}", RuleKind.COMPENSATION,
                            f"выполнено ребро {type_id}: {edge}", "откат невозможен (необратимо)",
                        )
                    )
                    continue
                # Структурная проверка: условия рёбер (guard) проверит исполнитель при откате.
                back = plan_path(fsm, edge.target, lambda s, src=edge.source: s == src, respect_guards=False)
                if back is None and edge.source != edge.target:
                    level = "опасное " if edge.risk == "high" else ""
                    warnings.append(f"type {type_id}: у {level}ребра {edge} нет пути отката")
                    continue
                how = " → ".join(e.event for e in back) if back else "восстановить прежнее значение"
                rules.append(
                    Rule(
                        f"comp:{type_id}:{edge.event}:{edge.source}", RuleKind.COMPENSATION,
                        f"выполнено ребро {type_id}: {edge}", f"откат: {how}",
                    )
                )


def scenario_rule(scenario: ScenarioSpec) -> Rule:
    steps = "; ".join(
        f"{s.action}({s.target.describe()}{'' if s.critical else ', некритично'})" for s in scenario.steps
    )
    kind = RuleKind.MINED if scenario.source == "mined" else RuleKind.SCENARIO
    return Rule(
        f"scn:{scenario.id}", kind, f"запрошен сценарий «{scenario.title}»", f"выполнить сагу: {steps}",
        scenario.source,
    )


def apply_overlay_item(ruleset: RuleSet, item: dict[str, Any]) -> None:
    """Добавляет синоним из оверлея (версионированное дополнение к спецификации)."""
    lexicon = ruleset.lexicon
    version = int(item["version"])
    report = lexicon.add(
        item["alias"], item["ref"], hint=item.get("hint"), weight=float(item.get("weight", 1.0)),
        source=f"overlay:v{version}", version=version,
    )
    if report.entry is not None:
        lexicon.changelog.append((version, f"+ «{report.entry.alias}» → {item['ref']}"))
        ruleset.rules.append(
            Rule(
                f"lex:{report.entry.id}", RuleKind.LEXICAL, f"во фразе есть «{report.entry.alias}»",
                item["ref"], f"overlay:v{version}", {"entry": report.entry.id},
            )
        )
    ruleset.overlay.append(dict(item))
