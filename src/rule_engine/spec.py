"""Декларативная спецификация: словарь команд, устройств, объектов и сценариев.

Спецификация — входной «набор», из которого :mod:`rule_engine.rules` составляет
набор правил. Формат — JSON (см. ``data/spec.json``); синонимы задаются строкой
или объектом ``{"alias": ..., "hint": ..., "weight": ...}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RISK_LEVELS = ("low", "medium", "high")


def risk_rank(risk: str) -> int:
    return RISK_LEVELS.index(risk)


def max_risk(*risks: str) -> str:
    return max(risks, key=risk_rank) if risks else "low"


class SpecError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("ошибки в спецификации:\n  - " + "\n  - ".join(problems))
        self.problems = problems


@dataclass(frozen=True, slots=True)
class AliasSpec:
    alias: str
    hint: str | None = None
    weight: float = 1.0

    @classmethod
    def parse(cls, raw: str | dict[str, Any]) -> AliasSpec:
        if isinstance(raw, str):
            return cls(raw)
        return cls(raw["alias"], raw.get("hint"), float(raw.get("weight", 1.0)))

    def to_json(self) -> str | dict[str, Any]:
        if self.hint is None and self.weight == 1.0:
            return self.alias
        out: dict[str, Any] = {"alias": self.alias}
        if self.hint is not None:
            out["hint"] = self.hint
        if self.weight != 1.0:
            out["weight"] = self.weight
        return out


def _aliases(raw: dict[str, Any]) -> tuple[AliasSpec, ...]:
    return tuple(AliasSpec.parse(a) for a in raw.get("aliases", ()))


@dataclass(frozen=True, slots=True)
class RoomSpec:
    id: str
    aliases: tuple[AliasSpec, ...]


@dataclass(frozen=True, slots=True)
class ActionSpec:
    id: str
    aliases: tuple[AliasSpec, ...]
    risk: str = "low"
    antonym: str | None = None
    compensation: str | None = None
    requires_value: bool = False
    target_kind: str = "device"  # "device" или "object" (объект сцены, исполняет манипулятор)
    performer: str | None = None  # тип устройства-исполнителя для target_kind=object
    frequency: float = 1.0
    title: str = ""


@dataclass(frozen=True, slots=True)
class TransitionSpec:
    action: str
    sources: tuple[str, ...]  # "*" = из любого состояния
    target: str  # "=" = остаться в том же состоянии (установка значения)
    value: str | None = None  # какое свойство меняется (temperature, brightness)
    range: tuple[float, float] | None = None
    risk: str | None = None
    cost: float = 1.0
    auto: bool = False
    guard: str | None = None
    irreversible: bool = False


@dataclass(frozen=True, slots=True)
class DeviceTypeSpec:
    id: str
    aliases: tuple[AliasSpec, ...]
    states: tuple[str, ...]
    initial: str
    transitions: tuple[TransitionSpec, ...]
    group_ok: bool = False
    timeout_ms: float = 2000.0
    latency_ms: float = 150.0
    title: str = ""


@dataclass(frozen=True, slots=True)
class SimSpec:
    latency_ms: float | None = None
    failure_rate: float = 0.0
    late_apply_rate: float = 0.3  # доля таймаутов, после которых команда всё же применилась
    online: bool = True


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    id: str
    type: str
    room: str | None
    aliases: tuple[AliasSpec, ...]
    name: str = ""
    state: str | None = None
    values: dict[str, float] = field(default_factory=dict)
    frequency: float = 1.0
    sim: SimSpec = field(default_factory=SimSpec)


@dataclass(frozen=True, slots=True)
class ObjectTypeSpec:
    id: str
    aliases: tuple[AliasSpec, ...]
    graspable: bool = True


@dataclass(frozen=True, slots=True)
class SceneObject:
    id: str
    type: str
    attrs: dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> SceneObject:
        return cls(raw["id"], raw["type"], dict(raw.get("attrs", {})), float(raw.get("confidence", 1.0)))


@dataclass(frozen=True, slots=True)
class TargetSelector:
    device: str | None = None
    type: str | None = None
    room: str | None = None  # None = любая комната

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> TargetSelector:
        return cls(raw.get("device"), raw.get("type"), raw.get("room"))

    def to_json(self) -> dict[str, str]:
        return {k: v for k, v in (("device", self.device), ("type", self.type), ("room", self.room)) if v}

    def describe(self) -> str:
        if self.device:
            return self.device
        return f"{self.type or '*'}@{self.room or '*'}"


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    action: str
    target: TargetSelector
    critical: bool = True
    value: float | None = None


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    id: str
    aliases: tuple[AliasSpec, ...]
    steps: tuple[ScenarioStep, ...]
    ordered: bool = False  # важен ли порядок шагов при распознавании в потоке
    title: str = ""
    source: str = "spec"
    support: float | None = None  # для выученных сценариев


@dataclass
class Spec:
    rooms: dict[str, RoomSpec]
    actions: dict[str, ActionSpec]
    device_types: dict[str, DeviceTypeSpec]
    devices: dict[str, DeviceSpec]
    object_types: dict[str, ObjectTypeSpec]
    attributes: dict[str, dict[str, tuple[AliasSpec, ...]]]
    scenarios: dict[str, ScenarioSpec]
    scene: list[SceneObject]
    policy: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- parsing
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Spec:
        rooms = {r["id"]: RoomSpec(r["id"], _aliases(r)) for r in data.get("rooms", ())}
        actions = {
            a["id"]: ActionSpec(
                id=a["id"],
                aliases=_aliases(a),
                risk=a.get("risk", "low"),
                antonym=a.get("antonym"),
                compensation=a.get("compensation"),
                requires_value=bool(a.get("requires_value", False)),
                target_kind=a.get("target_kind", "device"),
                performer=a.get("performer"),
                frequency=float(a.get("frequency", 1.0)),
                title=a.get("title", a["id"]),
            )
            for a in data.get("actions", ())
        }
        types = {}
        for t in data.get("device_types", ()):
            transitions = []
            for tr in t.get("transitions", ()):
                sources = tr.get("from", "*")
                transitions.append(
                    TransitionSpec(
                        action=tr["action"],
                        sources=tuple(t["states"]) if sources == "*" else tuple(sources),
                        target=tr["to"],
                        value=tr.get("value"),
                        range=tuple(tr["range"]) if "range" in tr else None,
                        risk=tr.get("risk"),
                        cost=float(tr.get("cost", 1.0)),
                        auto=bool(tr.get("auto", False)),
                        guard=tr.get("guard"),
                        irreversible=bool(tr.get("irreversible", False)),
                    )
                )
            types[t["id"]] = DeviceTypeSpec(
                id=t["id"],
                aliases=_aliases(t),
                states=tuple(t["states"]),
                initial=t.get("initial", t["states"][0]),
                transitions=tuple(transitions),
                group_ok=bool(t.get("group_ok", False)),
                timeout_ms=float(t.get("timeout_ms", 2000)),
                latency_ms=float(t.get("latency_ms", 150)),
                title=t.get("title", t["id"]),
            )
        devices = {}
        for d in data.get("devices", ()):
            sim = d.get("sim", {})
            devices[d["id"]] = DeviceSpec(
                id=d["id"],
                type=d["type"],
                room=d.get("room"),
                aliases=_aliases(d),
                name=d.get("name", d["id"]),
                state=d.get("state"),
                values={k: float(v) for k, v in d.get("values", {}).items()},
                frequency=float(d.get("frequency", 1.0)),
                sim=SimSpec(
                    latency_ms=sim.get("latency_ms"),
                    failure_rate=float(sim.get("failure_rate", 0.0)),
                    late_apply_rate=float(sim.get("late_apply_rate", 0.3)),
                    online=bool(sim.get("online", True)),
                ),
            )
        objects = {
            o["id"]: ObjectTypeSpec(o["id"], _aliases(o), bool(o.get("graspable", True)))
            for o in data.get("object_types", ())
        }
        attributes = {
            name: {value: tuple(AliasSpec.parse(a) for a in aliases) for value, aliases in values.items()}
            for name, values in data.get("attributes", {}).items()
        }
        scenarios = {s["id"]: parse_scenario(s) for s in data.get("scenarios", ())}
        scene = [SceneObject.parse(o) for o in data.get("scene", ())]
        spec = cls(
            rooms, actions, types, devices, objects, attributes, scenarios, scene,
            dict(data.get("policy", {})), dict(data.get("execution", {})), data,
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str | Path) -> Spec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -------------------------------------------------------------- validation
    def validate(self) -> None:
        problems: list[str] = []
        for a in self.actions.values():
            if a.risk not in RISK_LEVELS:
                problems.append(f"action {a.id}: неизвестный риск {a.risk!r}")
            for ref, label in ((a.antonym, "antonym"), (a.compensation, "compensation")):
                if ref and ref not in self.actions:
                    problems.append(f"action {a.id}: {label} {ref!r} не объявлен")
            if a.antonym and self.actions.get(a.antonym, a).antonym not in (None, a.id):
                problems.append(f"action {a.id}: антонимия должна быть взаимной")
            if a.target_kind == "object" and a.performer not in self.device_types:
                problems.append(f"action {a.id}: исполнитель {a.performer!r} не объявлен")
            for alias in a.aliases:
                if alias.hint and alias.hint not in self.device_types:
                    problems.append(f"action {a.id}: подсказка {alias.hint!r} — не тип устройства")
        for t in self.device_types.values():
            if t.initial not in t.states:
                problems.append(f"type {t.id}: начальное состояние {t.initial!r} не объявлено")
            for tr in t.transitions:
                if tr.action not in self.actions:
                    problems.append(f"type {t.id}: действие {tr.action!r} не объявлено")
                for s in (*tr.sources, tr.target):
                    if s != "=" and s not in t.states:
                        problems.append(f"type {t.id}: состояние {s!r} в {tr.action} не объявлено")
                if tr.risk and tr.risk not in RISK_LEVELS:
                    problems.append(f"type {t.id}: неизвестный риск {tr.risk!r}")
                action = self.actions.get(tr.action)
                if action and action.requires_value and not tr.value:
                    problems.append(f"type {t.id}: {tr.action} требует value")
        for d in self.devices.values():
            dtype = self.device_types.get(d.type)
            if dtype is None:
                problems.append(f"device {d.id}: тип {d.type!r} не объявлен")
            elif d.state and d.state not in dtype.states:
                problems.append(f"device {d.id}: состояние {d.state!r} не из {dtype.states}")
            if d.room and d.room not in self.rooms:
                problems.append(f"device {d.id}: комната {d.room!r} не объявлена")
        for s in self.scenarios.values():
            problems.extend(self._validate_scenario(s))
        for o in self.scene:
            if o.type not in self.object_types:
                problems.append(f"scene {o.id}: тип объекта {o.type!r} не объявлен")
        if problems:
            raise SpecError(problems)

    def _validate_scenario(self, s: ScenarioSpec) -> list[str]:
        problems = []
        for step in s.steps:
            if step.action not in self.actions:
                problems.append(f"scenario {s.id}: действие {step.action!r} не объявлено")
            t = step.target
            if t.device and t.device not in self.devices:
                problems.append(f"scenario {s.id}: устройство {t.device!r} не объявлено")
            if t.type and t.type not in self.device_types:
                problems.append(f"scenario {s.id}: тип {t.type!r} не объявлен")
            if t.room and t.room not in self.rooms:
                problems.append(f"scenario {s.id}: комната {t.room!r} не объявлена")
            if not (t.device or t.type):
                problems.append(f"scenario {s.id}: у шага {step.action} нет цели")
        return problems

    # ----------------------------------------------------------------- helpers
    def devices_of(self, type_id: str, room: str | None = None) -> list[DeviceSpec]:
        """Устройства типа (и комнаты) — по индексу, а не перебором всех устройств."""
        index = self.__dict__.get("_by_type_room")
        if index is None or self.__dict__.get("_indexed_count") != len(self.devices):
            index = {}
            for d in self.devices.values():
                index.setdefault((d.type, None), []).append(d)
                index.setdefault((d.type, d.room), []).append(d)
            self.__dict__["_by_type_room"] = index
            self.__dict__["_indexed_count"] = len(self.devices)
        return list(index.get((type_id, room), ()))

    def add_scenario(self, scenario: ScenarioSpec) -> None:
        problems = self._validate_scenario(scenario)
        if problems:
            raise SpecError(problems)
        self.scenarios[scenario.id] = scenario


def parse_scenario(raw: dict[str, Any]) -> ScenarioSpec:
    return ScenarioSpec(
        id=raw["id"],
        aliases=_aliases(raw),
        steps=tuple(
            ScenarioStep(
                st["action"],
                TargetSelector.parse(st["target"]),
                bool(st.get("critical", True)),
                st.get("value"),
            )
            for st in raw.get("steps", ())
        ),
        ordered=bool(raw.get("ordered", False)),
        title=raw.get("title", raw["id"]),
        source=raw.get("source", "spec"),
        support=raw.get("support"),
    )


def scenario_to_json(s: ScenarioSpec) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": s.id,
        "title": s.title,
        "aliases": [a.to_json() for a in s.aliases],
        "ordered": s.ordered,
        "source": s.source,
        "steps": [
            {
                "action": st.action,
                "target": st.target.to_json(),
                **({"critical": False} if not st.critical else {}),
                **({"value": st.value} if st.value is not None else {}),
            }
            for st in s.steps
        ],
    }
    if s.support is not None:
        out["support"] = round(s.support, 4)
    return out
