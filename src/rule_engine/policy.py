"""Политика решения: выполнить, уточнить, подтвердить или безопасно отказать.

Политика — упорядоченный список правил; срабатывает первое подходящее, и его
идентификатор попадает в объяснение. Главный инвариант: ни опечатка, ни
синоним не приводят к *молчаливому* выполнению опасной команды.

Безопасный отказ — любой исход, кроме выполнения неверной команды:
REJECT/NOT_FOUND (ничего не делаем), CLARIFY (варианты), CONFIRM (да/нет).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rule_engine.errors import EngineError, ErrorCode, Strategy
from rule_engine.resolver import Command, Interpretation
from rule_engine.rules import RuleSet
from rule_engine.spec import risk_rank
from rule_engine.text.distance import weighted_distance
from rule_engine.text.normalize import NormalizedQuery


class Status(StrEnum):
    OK = "OK"
    CLARIFY = "CLARIFY"
    CONFIRM = "CONFIRM"
    REJECT = "REJECT"
    NOT_FOUND = "NOT_FOUND"


@dataclass
class PolicyConfig:
    min_score: float = 0.55  # ниже — переспрашиваем
    reject_score: float = 0.35  # ниже — «не понял»
    margin: float = 0.06  # отрыв лидера, при котором вариант считается однозначным
    # насколько антоним должен быть дальше лучшего слова; 0.7 подобрано стресс-тестом
    # (evaluation.antonym_stress): минимальный порог с нулём молчаливых антонимов + запас
    antonym_margin: float = 0.7
    confirm_high_risk: bool = True  # подтверждать любую опасную команду
    fuzzy_max_risk: str = "medium"  # максимальный риск, при котором можно доверять опечатке
    max_options: int = 4
    rules_enabled: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyConfig:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def enabled(self, rule: str) -> bool:
        return self.rules_enabled.get(rule, True)


@dataclass
class Decision:
    status: Status
    chosen: Interpretation | None = None
    options: list[Interpretation] = field(default_factory=list)
    errors: list[EngineError] = field(default_factory=list)
    fired: list[str] = field(default_factory=list)
    question: str = ""

    @property
    def top(self) -> Interpretation | None:
        return self.chosen or (self.options[0] if self.options else None)


class Policy:
    """Упорядоченные правила решения. Каждое возвращает Decision или None."""

    def __init__(self, ruleset: RuleSet, config: PolicyConfig | None = None) -> None:
        self.ruleset = ruleset
        self.config = config or PolicyConfig()
        self.rules: list[tuple[str, str, Callable[..., Decision | None]]] = [
            ("P1-empty", "пустой запрос → отказ", self._empty),
            ("P2-negation", "отрицание → отказ, ничего не делаем", self._negation),
            ("P3-not-found", "нет гипотез или очень низкий скор → «не понял»", self._not_found),
            ("P4-hard-issues", "значение вне диапазона / несовместимо → отказ", self._hard_issues),
            ("P5-antonym", "нечёткий глагол близок к антониму → переспросить", self._antonym),
            ("P6-missing", "не хватает действия/цели/значения → переспросить", self._missing),
            ("P7-ambiguous", "второй вариант в пределах margin → переспросить", self._ambiguous),
            ("P8-low-confidence", "скор ниже порога → переспросить", self._low_confidence),
            ("P9-risk", "опасная команда → подтверждение", self._risk),
        ]

    def describe(self) -> list[str]:
        return [f"{rid}: {desc}" for rid, desc, _ in self.rules]

    def decide(self, query: NormalizedQuery, interpretations: list[Interpretation]) -> Decision:
        for rule_id, _, rule in self.rules:
            if not self.config.enabled(rule_id):
                continue
            decision = rule(query, interpretations)
            if decision is not None:
                decision.fired.append(rule_id)
                return decision
        return Decision(Status.OK, chosen=interpretations[0], fired=["P10-ok"])

    # ------------------------------------------------------------------ rules
    def _empty(self, query: NormalizedQuery, interps: list[Interpretation]) -> Decision | None:
        if not query.content_tokens and not interps:
            return Decision(Status.REJECT, errors=[EngineError(ErrorCode.EMPTY_INPUT)])
        return None

    def _negation(self, query: NormalizedQuery, interps: list[Interpretation]) -> Decision | None:
        if query.negated or (interps and any(c.negated for c in interps[0].commands)):
            return Decision(
                Status.REJECT, options=interps[:1], errors=[EngineError(ErrorCode.NEGATED)]
            )
        return None

    def _not_found(self, query: NormalizedQuery, interps: list[Interpretation]) -> Decision | None:
        if not interps:
            return Decision(Status.NOT_FOUND, errors=[EngineError(ErrorCode.NOT_FOUND)])
        top = interps[0]
        if top.score < self.config.reject_score:
            return Decision(
                Status.NOT_FOUND, options=interps[:1],
                errors=[EngineError(ErrorCode.LOW_CONFIDENCE, f"скор {top.score:.2f}")],
            )
        hard = [i for i in top.issues if i.code is ErrorCode.NOT_FOUND]
        if hard:
            return Decision(Status.NOT_FOUND, options=interps[:1], errors=hard)
        # Ни одного точного совпадения и есть нераспознанные слова: «какая погода
        # завтра» нечётко похоже на «погаси … закрой», но переспрашивать о таком не стоит.
        if not any(c.anchored for c in top.commands) and top.unknown:
            return Decision(
                Status.NOT_FOUND, options=interps[:1],
                errors=[EngineError(ErrorCode.LOW_CONFIDENCE, "нет ни одного точного совпадения")],
            )
        return None


    def _hard_issues(self, query: NormalizedQuery, interps: list[Interpretation]) -> Decision | None:
        top = interps[0]
        hard = [i for i in top.issues if i.strategy is Strategy.REJECT]
        if not hard:
            return None
        # Если есть равноценная гипотеза без жёстких проблем — пусть решают следующие правила.
        alternatives = [
            i for i in interps[1:]
            if top.score - i.score <= self.config.margin and not any(x.strategy is Strategy.REJECT for x in i.issues)
        ]
        if alternatives:
            return None
        return Decision(Status.REJECT, options=interps[:1], errors=hard)

    def _antonym(self, query: NormalizedQuery, interps: list[Interpretation]) -> Decision | None:
        top = interps[0]
        for command in top.commands:
            span = command.action_span
            if span is None or span.strict or command.action is None:
                continue
            antonym = self.ruleset.antonym(command.action)
            if antonym is None:
                continue
            best = span.cost
            words = [query.tokens[i].text for i in range(span.start, span.end)]
            ant_words = {w for a in self.ruleset.aliases_of(f"action:{antonym}") for w in a.split()}
            limit = best + self.config.antonym_margin
            d_ant = min(
                (weighted_distance(w, a, limit) for w in words for a in ant_words), default=float("inf")
            )
            if d_ant - best < self.config.antonym_margin:
                flipped = Interpretation(
                    tuple(
                        _replace_action(c, antonym) if c is command else c for c in top.commands
                    ),
                    top.score,
                    top.features,
                    top.explain + [f"антоним {antonym}: d={d_ant:.1f} против d={best:.1f}"],
                )
                return Decision(
                    Status.CLARIFY,
                    options=[top, flipped],
                    errors=[
                        EngineError(
                            ErrorCode.ANTONYM_CONFLICT,
                            f"«{' '.join(words)}»: {command.action} (d={best:.1f}) или {antonym} (d={d_ant:.1f})?",
                        )
                    ],
                )
        return None

    def _ambiguous(self, query: NormalizedQuery, interps: list[Interpretation]) -> Decision | None:
        top = interps[0]
        close = [
            i for i in interps
            if top.score - i.score <= self.config.margin
            and not any(x.strategy is Strategy.REJECT for x in i.issues)
        ]
        if len(close) < 2:
            return None
        # Одинаковые действия над одинаковыми целями с разным значением — тоже неоднозначность.
        return Decision(
            Status.CLARIFY,
            options=close,
            errors=[EngineError(ErrorCode.AMBIGUOUS, f"равноценных вариантов: {len(close)}")],
        )

    def _missing(self, query: NormalizedQuery, interps: list[Interpretation]) -> Decision | None:
        top = interps[0]
        missing = [
            i for i in top.issues
            if i.code in (ErrorCode.MISSING_ACTION, ErrorCode.MISSING_TARGET, ErrorCode.MISSING_VALUE)
        ]
        if not missing:
            return None
        options = [i for i in interps if not any(x.strategy is Strategy.REJECT for x in i.issues)]
        return Decision(Status.CLARIFY, options=options or interps[:1], errors=missing)

    def _low_confidence(self, query: NormalizedQuery, interps: list[Interpretation]) -> Decision | None:
        top = interps[0]
        if top.score >= self.config.min_score:
            return None
        return Decision(
            Status.CLARIFY,
            options=interps[: self.config.max_options],
            errors=[EngineError(ErrorCode.LOW_CONFIDENCE, f"скор {top.score:.2f}")],
        )

    def _risk(self, query: NormalizedQuery, interps: list[Interpretation]) -> Decision | None:
        top = interps[0]
        fuzzy_limit = risk_rank(self.config.fuzzy_max_risk)
        for command in top.commands:
            rank = risk_rank(command.risk)
            if not command.strict and rank > fuzzy_limit:
                return Decision(
                    Status.CONFIRM, chosen=top,
                    errors=[EngineError(ErrorCode.UNSAFE_FUZZY, _describe_span(command, query))],
                )
        if self.config.confirm_high_risk and any(c.risk == "high" for c in top.commands):
            return Decision(Status.CONFIRM, chosen=top, errors=[EngineError(ErrorCode.HIGH_RISK)])
        return None


def _replace_action(command: Command, action: str) -> Command:
    from dataclasses import replace

    return replace(command, action=action, action_span=None, issues=())


def _describe_span(command: Command, query: NormalizedQuery) -> str:
    span = command.action_span
    if span is None:
        return ""
    return span.describe(query.tokens)
