"""Состояние мира глазами движка: состояния устройств, значения, сцена.

Это *убеждение* движка, а не физическая реальность: реальное состояние читает
драйвер, и при расхождении (STATE_DRIFT) мир синхронизируется с устройством.
Все переходы идут через :class:`~rule_engine.fsm.FSMInstance`, поэтому мир не
может оказаться в состоянии, недостижимом по рёбрам автомата.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from rule_engine.fsm import FSMInstance
from rule_engine.rules import RuleSet
from rule_engine.spec import SceneObject


@dataclass
class DeviceState:
    machine: FSMInstance
    values: dict[str, float] = field(default_factory=dict)
    online: bool = True
    holding: str | None = None  # для манипулятора: id объекта в захвате

    @property
    def state(self) -> str:
        return self.machine.state


class World:
    def __init__(self, ruleset: RuleSet, clock: Callable[[], float] | None = None) -> None:
        self.ruleset = ruleset
        self.clock = clock or (lambda: 0.0)
        self.devices: dict[str, DeviceState] = {}
        for device in ruleset.spec.devices.values():
            fsm = ruleset.fsms[device.type]
            self.devices[device.id] = DeviceState(
                FSMInstance(fsm, device.state or fsm.initial, self.clock),
                dict(device.values),
                device.sim.online,
            )
        self.scene: dict[str, SceneObject] = {o.id: o for o in ruleset.spec.scene}
        self.version = 0

    # ----------------------------------------------------------------- queries
    def state(self, device_id: str) -> str:
        return self.devices[device_id].state

    def value(self, device_id: str, prop: str) -> float | None:
        return self.devices[device_id].values.get(prop)

    def type_of(self, device_id: str) -> str:
        return self.ruleset.spec.devices[device_id].type

    def room_of(self, device_id: str) -> str | None:
        return self.ruleset.spec.devices[device_id].room

    def guard_context(self, device_id: str, obj: str | None = None) -> dict[str, Any]:
        ctx: dict[str, Any] = {"device": device_id, "holding": self.devices[device_id].holding}
        if obj is not None and obj in self.scene:
            scene_obj = self.scene[obj]
            ctx["object"] = obj
            ctx["object_confidence"] = scene_obj.confidence
            spec = self.ruleset.spec.object_types.get(scene_obj.type)
            ctx["object_graspable"] = bool(spec and spec.graspable)
        return ctx

    # ---------------------------------------------------------------- updates
    def apply(
        self,
        device_id: str,
        event: str,
        prop: str | None = None,
        value: float | None = None,
        obj: str | None = None,
        note: str = "",
    ) -> None:
        dev = self.devices[device_id]
        dev.machine.fire(event, self.guard_context(device_id, obj), note)
        if prop is not None and value is not None:
            dev.values[prop] = value
        if event == "grasp":
            dev.holding = obj
        elif event == "release":
            dev.holding = None
        self.version += 1

    def sync(self, device_id: str, state: str, values: dict[str, float], note: str) -> None:
        dev = self.devices[device_id]
        if dev.state != state:
            dev.machine.force(state, note)
        dev.values.update(values)
        self.version += 1

    def set_online(self, device_id: str, online: bool) -> None:
        self.devices[device_id].online = online

    def update_scene(self, objects: Iterable[SceneObject]) -> None:
        held = {d.holding for d in self.devices.values() if d.holding}
        fresh = {o.id: o for o in objects}
        # Объект в захвате манипулятора детектор может не видеть — он не исчезает.
        for obj_id in held:
            if obj_id not in fresh and obj_id in self.scene:
                fresh[obj_id] = self.scene[obj_id]
        self.scene = fresh
        self.version += 1

    # ------------------------------------------------------------- snapshots
    def snapshot(self) -> dict[str, Any]:
        return {
            "devices": {
                d: (s.state, copy.deepcopy(s.values), s.online, s.holding) for d, s in self.devices.items()
            },
            "scene": dict(self.scene),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        for device_id, (state, values, online, holding) in snap["devices"].items():
            dev = self.devices[device_id]
            if dev.state != state:
                dev.machine.force(state, "restore")
            dev.values = copy.deepcopy(values)
            dev.online, dev.holding = online, holding
        self.scene = dict(snap["scene"])
        self.version += 1

    def describe(self, device_id: str) -> str:
        dev = self.devices[device_id]
        extra = ", ".join(f"{k}={v:g}" for k, v in sorted(dev.values.items()))
        flags = [] if dev.online else ["offline"]
        if dev.holding:
            flags.append(f"держит {dev.holding}")
        tail = "; ".join(x for x in (extra, ", ".join(flags)) if x)
        return f"{dev.state}" + (f" ({tail})" if tail else "")
