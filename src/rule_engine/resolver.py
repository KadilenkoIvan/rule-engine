"""Сборка гипотез: из разбора — конкретные команды «действие + устройства/объекты».

Для каждой подкоманды перечисляются варианты целей:
  * явно названное устройство («люстра»);
  * тип устройства («свет») → устройства этого типа с учётом комнаты из запроса,
    комнаты-контекста (где стоит колонка) и квантора «все»;
  * объект сцены («красная чашка») → объекты от детектора, отфильтрованные
    по атрибутам; исполнитель — манипулятор;
  * цель не названа → устройства, к которым применимо действие.

Скоринг — явная линейная формула, каждое слагаемое попадает в объяснение:

    score = 0.45·качество_глагола + 0.25·качество_цели + 0.2·покрытие + 0.1·частота − штрафы
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice, product

from rule_engine.errors import EngineError, ErrorCode
from rule_engine.index.lexicon import STRICT_MATCH_TYPES, Kind, SpanMatch
from rule_engine.parser import Clause, Parse
from rule_engine.rules import RuleSet
from rule_engine.spec import max_risk
from rule_engine.text.normalize import NormalizedQuery
from rule_engine.world import World

W_ACTION, W_TARGET, W_COVERAGE, W_PRIOR = 0.45, 0.25, 0.2, 0.1
ROOM_MISMATCH_PENALTY = 0.15
PARSE_PENALTY = 0.12
UNUSED_SPAN_PENALTY = 0.1
PARSE_PENALTY_CAP = 0.3
PARSE_PRUNE = PARSE_PENALTY_CAP / PARSE_PENALTY  # дальше штраф уже не растёт
MAX_INFERRED = 16
MISSING_ROOM_DEVICE_PENALTY = 0.3
MAX_PRODUCT = 24


@dataclass(frozen=True, slots=True)
class Command:
    action: str | None
    targets: tuple[str, ...] = ()  # устройства (для объектов — исполнитель)
    objects: tuple[str, ...] = ()  # объекты сцены
    scenario: str | None = None
    value: float | None = None
    negated: bool = False
    risk: str = "low"
    action_span: SpanMatch | None = None
    issues: tuple[EngineError, ...] = ()
    notes: tuple[str, ...] = ()
    anchored: bool = True  # есть ли в подкоманде хоть одно точное (не нечёткое) совпадение

    def key(self) -> tuple:
        return (self.action, self.scenario, tuple(sorted(self.targets)), self.objects, self.value)

    @property
    def strict(self) -> bool:
        return self.action_span is None or self.action_span.strict


@dataclass
class ClauseOption:
    command: Command
    score: float
    features: dict[str, float]


@dataclass
class Interpretation:
    commands: tuple[Command, ...]
    score: float
    features: list[dict[str, float]]
    explain: list[str] = field(default_factory=list)
    unknown: tuple[str, ...] = ()  # слова запроса, которые разбор пропустил

    def key(self) -> tuple:
        return tuple(c.key() for c in self.commands)

    @property
    def issues(self) -> list[EngineError]:
        return [issue for c in self.commands for issue in c.issues]

    @property
    def risk(self) -> str:
        return max_risk(*(c.risk for c in self.commands))


@dataclass
class _TargetOption:
    devices: tuple[str, ...]
    objects: tuple[str, ...] = ()
    quality: float = 1.0
    penalty: float = 0.0
    notes: tuple[str, ...] = ()
    issues: tuple[EngineError, ...] = ()


def _anchored(spans: list[SpanMatch]) -> bool:
    return any(w.match in STRICT_MATCH_TYPES for s in spans for w in s.words)


def _hard(command: Command) -> bool:
    return any(i.code in (ErrorCode.VALUE_OUT_OF_RANGE, ErrorCode.INCOMPATIBLE) for i in command.issues)


def _add_issue(option: ClauseOption, issue: EngineError) -> ClauseOption:
    from dataclasses import replace

    if any(i.code is issue.code for i in option.command.issues):
        return option
    return ClauseOption(replace(option.command, issues=option.command.issues + (issue,)), option.score, option.features)


class Resolver:
    def __init__(self, ruleset: RuleSet, max_interpretations: int = 12) -> None:
        self.ruleset = ruleset
        self._parse_penalty = 0.0
        self.spec = ruleset.spec
        self.max_interpretations = max_interpretations
        freqs = [d.frequency for d in self.spec.devices.values()] or [1.0]
        self._log_max_freq = math.log1p(max(freqs))
        self._log_freq = {d.id: math.log1p(d.frequency) for d in self.spec.devices.values()}

    # ----------------------------------------------------------------- public
    def resolve(
        self,
        parses: list[Parse],
        query: NormalizedQuery,
        world: World,
        context_room: str | None = None,
    ) -> list[Interpretation]:
        best: dict[tuple, Interpretation] = {}
        top_parse = max((p.score for p in parses), default=0.0)
        for parse in parses:
            if top_parse - parse.score > PARSE_PRUNE:
                continue  # разбор заведомо хуже: штраф не дал бы ему победить
            # Разбор хуже лучшего (пропустил распознанное слово) — штраф пропорционально отставанию.
            self._parse_penalty = min(PARSE_PENALTY_CAP, PARSE_PENALTY * (top_parse - parse.score))
            per_clause = [self._clause(c, parse, world, context_room) for c in parse.clauses]
            per_clause = [opts for opts in per_clause if opts]
            if not per_clause:
                continue
            for combo in islice(product(*per_clause), MAX_PRODUCT):
                score = sum(o.score for o in combo) / len(combo)
                interp = Interpretation(
                    tuple(o.command for o in combo), round(score, 4), [o.features for o in combo],
                    unknown=tuple(parse.unknown),
                )
                interp.explain = self._explain(interp, parse, query)
                key = interp.key()
                if key not in best or interp.score > best[key].score:
                    best[key] = interp
        ranked = sorted(best.values(), key=lambda i: (-i.score, i.key().__repr__()))
        return ranked[: self.max_interpretations]

    # ---------------------------------------------------------------- clauses
    def _clause(
        self, clause: Clause, parse: Parse, world: World, context_room: str | None
    ) -> list[ClauseOption]:
        if clause.scenario is not None:
            return [self._scenario_option(clause, parse)]

        if clause.action is not None:
            action_id = clause.action.entry.ref.split(":", 1)[1]
            choices = [(action_id, clause.action, clause.action.entry.hint, ())]
        else:
            choices = self._guess_actions(clause, world, context_room)
        options: list[ClauseOption] = []
        for action_id, span, hint, extra_issues in choices:
            targets = self._targets(action_id, hint, clause, world, context_room)
            built = [self._option(action_id, span, t, clause, parse, extra_issues) for t in targets]
            if self._inferred(action_id, clause):
                # Цель выведена: варианты с жёсткими проблемами (не тот диапазон,
                # не та комната) отбрасываем, если есть чистые.
                clean = [o for o in built if not _hard(o.command)]
                built = clean or built
                if len(built) > 1 and not hint:
                    built = [_add_issue(o, EngineError(ErrorCode.MISSING_TARGET)) for o in built]
            options += built
        return options

    def _inferred(self, action_id: str | None, clause: Clause) -> bool:
        if action_id is None or self.spec.actions[action_id].target_kind == "object":
            return False
        return not any(s.entry.kind in (Kind.DEVICE, Kind.DEVICE_TYPE) for s in clause.targets)

    def _guess_actions(
        self, clause: Clause, world: World, context_room: str | None
    ) -> list[tuple[str | None, None, None, tuple[EngineError, ...]]]:
        """Глагола нет: предлагаем действия, применимые к названной цели."""
        if not clause.targets:
            return []
        types: set[str] = set()
        for option in self._targets(None, None, clause, world, context_room):
            types.update(self.spec.devices[d].type for d in option.devices)
        wants_value = bool(clause.values)
        actions = sorted(
            {
                a
                for t in types
                for a in self.ruleset.fsms[t].events()
                if self.spec.actions[a].requires_value == wants_value
                and self.spec.actions[a].target_kind == "device"
            }
        )
        # Число названо и ему подходит ровно одно действие («тёплый пол 30 градусов») —
        # действие однозначно выводится; иначе надо спросить, что сделать.
        issue = () if wants_value and len(actions) == 1 else (EngineError(ErrorCode.MISSING_ACTION),)
        return [(a, None, None, issue) for a in actions]

    def _scenario_option(self, clause: Clause, parse: Parse) -> ClauseOption:
        span = clause.scenario
        assert span is not None
        scenario_id = span.entry.ref.split(":", 1)[1]
        scenario = self.spec.scenarios[scenario_id]
        risks = []
        for step in scenario.steps:
            types = [step.target.type] if step.target.type else [self.spec.devices[step.target.device].type]
            risks += [self.ruleset.max_action_risk(step.action, t) for t in types if t in self.ruleset.fsms]
        quality = span.quality * span.entry.weight
        features = {
            "action": quality, "target": quality, "coverage": parse.coverage, "prior": 0.5,
            "penalty": round(self._parse_penalty, 4),
        }
        score = self._combine(features)
        command = Command(
            action=None, scenario=scenario_id, negated=clause.negated, risk=max_risk(*risks),
            action_span=span, notes=(f"сценарий «{scenario.title}»: {len(scenario.steps)} шагов",),
            anchored=_anchored([span]),
        )
        return ClauseOption(command, score, features)

    def _option(
        self,
        action_id: str | None,
        span: SpanMatch | None,
        target: _TargetOption,
        clause: Clause,
        parse: Parse,
        extra_issues: tuple[EngineError, ...],
    ) -> ClauseOption:
        issues = list(extra_issues) + list(target.issues)
        notes = list(target.notes)
        value = clause.values[0] if clause.values else None
        risk = "low"
        if action_id is not None:
            action = self.spec.actions[action_id]
            types = {self.spec.devices[d].type for d in target.devices}
            if action.requires_value:
                if value is None:
                    issues.append(EngineError(ErrorCode.MISSING_VALUE, f"({action.title})"))
                else:
                    issues += self._range_issues(action_id, types, value)
            elif value is not None:
                notes.append(f"число {value:g} проигнорировано: «{action.title}» не принимает значение")
                value = None
            incompatible = sorted(t for t in types if not self.ruleset.applicable(action_id, t))
            if incompatible:
                issues.append(
                    EngineError(
                        ErrorCode.INCOMPATIBLE,
                        f"«{action.title}» не применимо к: {', '.join(incompatible)}",
                        {"types": incompatible},
                    )
                )
            risk = max_risk(
                action.risk,
                *(self.ruleset.max_action_risk(action_id, t) for t in types if t in self.ruleset.fsms),
            )
        action_quality = span.quality * span.entry.weight if span is not None else 0.4
        features = {
            "action": round(action_quality, 4),
            "target": round(target.quality, 4),
            "coverage": round(parse.coverage, 4),
            "prior": round(self._prior(target.devices, action_id), 4),
            "penalty": round(target.penalty + self._parse_penalty, 4),
        }
        command = Command(
            action=action_id,
            targets=target.devices,
            objects=target.objects,
            value=value,
            negated=clause.negated,
            risk=risk,
            action_span=span,
            issues=tuple(issues),
            notes=tuple(notes),
            anchored=_anchored(clause.spans()),
        )
        return ClauseOption(command, self._combine(features), features)

    def _range_issues(self, action_id: str, types: set[str], value: float) -> list[EngineError]:
        issues = []
        for type_id in sorted(types):
            for edge in self.ruleset.fsms[type_id].edges_for(action_id):
                bounds = edge.meta.get("range")
                if bounds and not bounds[0] <= value <= bounds[1]:
                    issues.append(
                        EngineError(
                            ErrorCode.VALUE_OUT_OF_RANGE,
                            f"{value:g} для {type_id}: допустимо {bounds[0]:g}–{bounds[1]:g}",
                            {"range": list(bounds)},
                        )
                    )
                    break
        return issues

    @staticmethod
    def _combine(f: dict[str, float]) -> float:
        score = (
            W_ACTION * f["action"]
            + W_TARGET * f["target"]
            + W_COVERAGE * f["coverage"]
            + W_PRIOR * f["prior"]
            - f.get("penalty", 0.0)
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _prior(self, devices: Iterable[str], action_id: str | None) -> float:
        devices = list(devices)
        if not devices:
            return 0.3
        dev = sum(self._log_freq.get(d, 0.0) for d in devices) / len(devices)
        prior = dev / self._log_max_freq if self._log_max_freq else 0.5
        return min(1.0, prior)

    # ----------------------------------------------------------------- targets
    def _rooms(self, clause: Clause) -> list[str]:
        return list(dict.fromkeys(s.entry.ref.split(":", 1)[1] for s in clause.rooms))

    def _targets(
        self,
        action_id: str | None,
        hint: str | None,
        clause: Clause,
        world: World,
        context_room: str | None,
    ) -> list[_TargetOption]:
        action = self.spec.actions.get(action_id) if action_id else None
        rooms = self._rooms(clause)
        if action is not None and action.target_kind == "object":
            return self._object_targets(action_id, clause, rooms, world, context_room)

        groups: list[list[_TargetOption]] = []
        room_quality = [s.quality for s in clause.rooms]
        for span in clause.targets:
            ref_kind, ref_id = span.entry.ref.split(":", 1)
            quality = (span.quality * span.entry.weight + sum(room_quality)) / (1 + len(room_quality))
            if span.entry.kind is Kind.DEVICE:
                device = self.spec.devices[ref_id]
                penalty, notes = 0.0, [f"устройство: {device.name}"]
                extra_rooms = [r for r in rooms if r != device.room]
                if extra_rooms and device.room and self._alias_names_room(span, device.room):
                    # «свет на кухне и в прихожей»: синоним устройства сам содержит комнату,
                    # значит, остальные комнаты относятся к тому же типу устройств.
                    groups.append(
                        self._type_options(
                            device.type, [device.room, *extra_rooms], True, context_room, quality,
                            notes=(f"«{span.entry.alias}» + комнаты {extra_rooms}",),
                        )
                    )
                    continue
                if rooms and device.room not in rooms:
                    penalty = ROOM_MISMATCH_PENALTY
                    notes.append(f"{device.name} находится в «{device.room}», а не в {rooms}")
                groups.append([_TargetOption((ref_id,), (), quality, penalty, tuple(notes))])
            elif span.entry.kind is Kind.DEVICE_TYPE:
                groups.append(self._type_options(ref_id, rooms, clause.quantifier, context_room, quality))
            # объекты сцены для «устройственных» действий — лишь уточнение
        if not groups:
            groups = [self._inferred_options(action_id, hint, rooms, clause, context_room)]
        options: list[_TargetOption] = []
        for combo in islice(product(*groups), MAX_PRODUCT):
            devices = tuple(dict.fromkeys(d for option in combo for d in option.devices))
            if not devices:
                continue
            options.append(
                _TargetOption(
                    devices,
                    (),
                    sum(o.quality for o in combo) / len(combo),
                    max(o.penalty for o in combo),
                    tuple(n for o in combo for n in o.notes),
                    tuple(i for o in combo for i in o.issues),
                )
            )
        return options

    def _alias_names_room(self, span: SpanMatch, room: str) -> bool:
        room_words = {w for a in self.spec.rooms[room].aliases for w in a.alias.split()}
        return bool(room_words & set(span.entry.tokens))

    def _type_options(
        self,
        type_id: str,
        rooms: list[str],
        quantifier: bool,
        context_room: str | None,
        quality: float,
        penalty: float = 0.0,
        notes: tuple[str, ...] = (),
        limit: int | None = None,
    ) -> list[_TargetOption]:
        dtype = self.spec.device_types[type_id]
        all_devices = [d.id for d in self.spec.devices_of(type_id)]
        if not all_devices:
            return []
        notes = notes + (f"тип: {dtype.title}",)
        if rooms:
            devices = [d for d in all_devices if self.spec.devices[d].room in rooms]
            if not devices:
                devices = all_devices
                penalty += MISSING_ROOM_DEVICE_PENALTY
                notes += (f"в {rooms} нет устройства типа {type_id}",)
            else:
                notes += (f"комната: {', '.join(rooms)} (из запроса)",)
        elif context_room and any(self.spec.devices[d].room == context_room for d in all_devices):
            devices = [d for d in all_devices if self.spec.devices[d].room == context_room]
            notes += (f"комната: {context_room} (контекст)",)
        else:
            devices = all_devices
        if quantifier:
            return [_TargetOption(tuple(devices), (), quality, penalty, notes + ("все устройства (квантор)",))]
        if len(devices) == 1:
            return [_TargetOption(tuple(devices), (), quality, penalty, notes)]
        if dtype.group_ok:
            by_room: dict[str | None, list[str]] = defaultdict(list)
            for d in devices:
                by_room[self.spec.devices[d].room].append(d)
            if rooms or len(by_room) == 1:
                return [_TargetOption(tuple(devices), (), quality, penalty, notes + ("групповая команда",))]
            return [
                _TargetOption(tuple(group), (), quality, penalty, notes + (f"вариант: {room}",))
                for room, group in sorted(by_room.items(), key=lambda kv: str(kv[0]))
            ]
        if limit is not None and len(devices) > limit:
            devices = sorted(devices, key=lambda d: -self._log_freq.get(d, 0.0))[:limit]
        return [
            _TargetOption((d,), (), quality, penalty, notes + (f"вариант: {self.spec.devices[d].name}",))
            for d in devices
        ]

    def _inferred_options(
        self,
        action_id: str | None,
        hint: str | None,
        rooms: list[str],
        clause: Clause,
        context_room: str | None,
    ) -> list[_TargetOption]:
        """Цель не названа: подсказка синонима или все применимые типы."""
        if action_id is None:
            return []
        if hint:
            return self._type_options(
                hint, rooms, clause.quantifier, context_room, 0.85, notes=(f"цель из синонима глагола: {hint}",)
            )
        types = sorted(self.ruleset.types_for_action(action_id))
        if rooms:
            in_rooms = [t for t in types if self.spec.devices_of(t) and any(
                self.spec.devices[d.id].room in rooms for d in self.spec.devices_of(t))]
            types = in_rooms or types
        options: list[_TargetOption] = []
        for type_id in types:
            for option in self._type_options(
                type_id, rooms, clause.quantifier, context_room, 0.55, limit=MAX_INFERRED
            ):
                option.notes += ("цель не названа — выведена по применимости",)
                options.append(option)
        # Цель не названа — это всё равно будет уточнение; длинный хвост вариантов не нужен.
        if len(options) > MAX_INFERRED:
            options.sort(key=lambda o: -self._prior(o.devices, action_id))
            options = options[:MAX_INFERRED]
        return options

    def _object_targets(
        self,
        action_id: str,
        clause: Clause,
        rooms: list[str],
        world: World,
        context_room: str | None,
    ) -> list[_TargetOption]:
        action = self.spec.actions[action_id]
        performers = [d.id for d in self.spec.devices_of(action.performer or "")]
        named = [s.entry.ref.split(":", 1)[1] for s in clause.targets if s.entry.kind is Kind.DEVICE]
        if named:
            performers = [p for p in performers if p in named] or performers
        if rooms and any(self.spec.devices[p].room in rooms for p in performers):
            performers = [p for p in performers if self.spec.devices[p].room in rooms]
        if not performers:
            return [
                _TargetOption((), (), 0.0, 0.0, (), (EngineError(ErrorCode.INCOMPATIBLE, "нет исполнителя"),))
            ]
        type_spans = [s for s in clause.targets if s.entry.kind is Kind.OBJECT_TYPE]
        # Названные устройства, не являющиеся исполнителем, — неиспользованная улика.
        unused = [
            s for s in clause.targets
            if s.entry.kind is Kind.DEVICE_TYPE
            or (s.entry.kind is Kind.DEVICE and s.entry.ref.split(":", 1)[1] not in performers)
        ]
        unused_penalty = UNUSED_SPAN_PENALTY * len(unused)
        wanted_types = {s.entry.ref.split(":", 1)[1] for s in type_spans}
        wanted_attrs = [tuple(s.entry.ref.split(":", 1)[1].split("=", 1)) for s in clause.attrs]
        spans = type_spans + list(clause.attrs)
        quality = sum(s.quality for s in spans) / len(spans) if spans else 0.5

        objects = [
            o for o in world.scene.values()
            if (not wanted_types or o.type in wanted_types)
            and all(o.attrs.get(name) == value for name, value in wanted_attrs)
        ]
        described = " ".join(
            [f"{n}={v}" for n, v in wanted_attrs] + sorted(wanted_types)
        ) or "объект"
        if not objects:
            issue = EngineError(ErrorCode.NOT_FOUND, f"в поле зрения нет: {described}")
            return [_TargetOption((performers[0],), (), quality, 0.0, (), (issue,))]
        options = []
        issues = () if (wanted_types or wanted_attrs) else (EngineError(ErrorCode.MISSING_TARGET),)
        for performer in performers:
            for obj in sorted(objects, key=lambda o: -o.confidence):
                attrs = ", ".join(f"{k}={v}" for k, v in sorted(obj.attrs.items()))
                options.append(
                    _TargetOption(
                        (performer,), (obj.id,), quality * (0.5 + 0.5 * obj.confidence), unused_penalty,
                        (f"объект {obj.id} ({obj.type}; {attrs}; уверенность {obj.confidence:.2f})",),
                        issues,
                    )
                )
        return options

    # ------------------------------------------------------------- explain
    def _explain(self, interp: Interpretation, parse: Parse, query: NormalizedQuery) -> list[str]:
        lines = []
        for clause_idx, (command, features) in enumerate(zip(interp.commands, interp.features, strict=True)):
            prefix = f"[{clause_idx + 1}] " if len(interp.commands) > 1 else ""
            if command.action_span is not None:
                lines.append(prefix + "глагол: " + command.action_span.describe(query.tokens))
            elif command.action:
                lines.append(prefix + f"глагол не назван — вариант «{command.action}»")
            for clause in parse.clauses:
                for span in clause.targets + clause.rooms + clause.attrs:
                    line = prefix + "цель: " + span.describe(query.tokens)
                    if line not in lines:
                        lines.append(line)
            lines += [prefix + note for note in command.notes]
            formula = " + ".join(
                f"{w}·{k}={features.get(k, 0):.2f}"
                for w, k in ((W_ACTION, "action"), (W_TARGET, "target"), (W_COVERAGE, "coverage"), (W_PRIOR, "prior"))
            )
            if features.get("penalty"):
                formula += f" − штраф {features['penalty']:.2f}"
            lines.append(prefix + f"скоринг: {formula}")
        if parse.unknown:
            lines.append("не распознаны слова: " + ", ".join(parse.unknown))
        return lines
