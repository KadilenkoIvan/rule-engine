"""Конечные автоматы: определение, экземпляр, планирование пути по рёбрам.

Один и тот же класс используется в двух местах:

* автомат *устройства* (лампа: off ↔ on; манипулятор: parked → ready → holding),
  его рёбра помечены действиями (``turn_on``, ``grasp``) и имеют стоимость и риск;
* автомат *обработки запроса* (RECEIVED → … → SUCCEEDED / ROLLED_BACK), который
  гарантирует, что конвейер не перепрыгнет через этап.

«Как ходить по ребру»: переход разрешён, только если ребро существует из
текущего состояния и его guard истинен; иначе — :class:`TransitionError`.
Если нужное ребро из текущего состояния недоступно, :func:`plan_path` ищет
кратчайший путь до состояния, откуда ребро доступно (DP Беллмана–Форда).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

Guard = Callable[[Mapping[str, Any]], bool]


class TransitionError(Exception):
    def __init__(self, machine: str, state: str, event: str, reason: str) -> None:
        super().__init__(f"{machine}: нельзя выполнить «{event}» из состояния «{state}»: {reason}")
        self.machine, self.state, self.event, self.reason = machine, state, event, reason


@dataclass(frozen=True, slots=True)
class Edge:
    event: str
    source: str
    target: str
    cost: float = 1.0
    risk: str = "low"
    auto: bool = False  # можно ли планировщику вставлять ребро как промежуточный шаг
    guard: Guard | None = None
    guard_name: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    def allowed(self, context: Mapping[str, Any]) -> bool:
        return self.guard is None or bool(self.guard(context))

    def __str__(self) -> str:
        return f"{self.source} --{self.event}--> {self.target}"


class FSMDefinition:
    def __init__(
        self,
        name: str,
        states: Iterable[str],
        initial: str,
        edges: Iterable[Edge],
        terminal: Iterable[str] = (),
    ) -> None:
        self.name = name
        self.states = tuple(states)
        self.initial = initial
        self.terminal = frozenset(terminal)
        self.edges = tuple(edges)
        known = set(self.states)
        if initial not in known:
            raise ValueError(f"{name}: начальное состояние {initial!r} не объявлено")
        self._out: dict[str, list[Edge]] = {s: [] for s in self.states}
        for edge in self.edges:
            for state in (edge.source, edge.target):
                if state not in known:
                    raise ValueError(f"{name}: ребро {edge} ссылается на неизвестное состояние {state!r}")
            self._out[edge.source].append(edge)

    def out_edges(self, state: str) -> list[Edge]:
        return self._out.get(state, [])

    def find(self, state: str, event: str) -> list[Edge]:
        return [e for e in self.out_edges(state) if e.event == event]

    def events(self) -> set[str]:
        return {e.event for e in self.edges}

    def edges_for(self, event: str) -> list[Edge]:
        return [e for e in self.edges if e.event == event]

    def reachable(self, start: str | None = None) -> set[str]:
        seen = {start or self.initial}
        stack = [start or self.initial]
        while stack:
            for edge in self.out_edges(stack.pop()):
                if edge.target not in seen:
                    seen.add(edge.target)
                    stack.append(edge.target)
        return seen

    def unreachable(self) -> set[str]:
        return set(self.states) - self.reachable()


def plan_path(
    fsm: FSMDefinition,
    start: str,
    goal: Callable[[str], bool],
    context: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
    edge_filter: Callable[[Edge], bool] = lambda e: True,
    respect_guards: bool = True,
) -> list[Edge] | None:
    """Кратчайший по стоимости путь из ``start`` в любое состояние, где ``goal`` истинно.

    DP Беллмана–Форда по числу рёбер: ``best[k][s]`` — минимальная стоимость
    дойти до ``s`` не более чем за ``k`` шагов:

        best[k][s] = min(best[k-1][s], min_{e: u→s} best[k-1][u] + cost(e))

    Ограничение ``max_steps`` естественно ложится на DP (длина плана ограничена),
    а предшественники позволяют восстановить сам путь.
    """
    context = context or {}
    steps = max_steps if max_steps is not None else len(fsm.states)
    best: dict[str, float] = {s: math.inf for s in fsm.states}
    best[start] = 0.0
    parent: dict[str, Edge | None] = {start: None}
    for _ in range(steps):
        updated = False
        layer = dict(best)
        for edge in fsm.edges:
            if not edge_filter(edge) or (respect_guards and not edge.allowed(context)):
                continue
            if best[edge.source] + edge.cost < layer[edge.target]:
                layer[edge.target] = best[edge.source] + edge.cost
                parent[edge.target] = edge
                updated = True
        best = layer
        if not updated:
            break
    targets = [s for s in fsm.states if goal(s) and best[s] < math.inf]
    if not targets:
        return None
    target = min(targets, key=lambda s: (best[s], s != start))
    path: list[Edge] = []
    seen: set[str] = set()
    while (edge := parent.get(target)) is not None and target != start:
        if target in seen:  # защитный случай; при неотрицательных весах не бывает
            return None
        seen.add(target)
        path.append(edge)
        target = edge.source
    path.reverse()
    return path


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    at: float
    source: str
    event: str
    target: str
    note: str = ""


class FSMInstance:
    """Экземпляр автомата с текущим состоянием, хуками и историей переходов."""

    def __init__(
        self,
        definition: FSMDefinition,
        state: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.definition = definition
        self.state = state or definition.initial
        if self.state not in definition.states:
            raise ValueError(f"{definition.name}: неизвестное состояние {self.state!r}")
        self.history: list[HistoryRecord] = []
        self._clock = clock or (lambda: 0.0)
        self._on_enter: dict[str, list[Callable[[HistoryRecord], None]]] = {}

    def on_enter(self, state: str, callback: Callable[[HistoryRecord], None]) -> None:
        self._on_enter.setdefault(state, []).append(callback)

    @property
    def finished(self) -> bool:
        return self.state in self.definition.terminal

    def can(self, event: str, context: Mapping[str, Any] | None = None) -> bool:
        return any(e.allowed(context or {}) for e in self.definition.find(self.state, event))

    def fire(self, event: str, context: Mapping[str, Any] | None = None, note: str = "") -> Edge:
        candidates = self.definition.find(self.state, event)
        if not candidates:
            raise TransitionError(self.definition.name, self.state, event, "нет такого ребра")
        allowed = [e for e in candidates if e.allowed(context or {})]
        if not allowed:
            names = ", ".join(e.guard_name or "guard" for e in candidates)
            raise TransitionError(self.definition.name, self.state, event, f"не выполнено условие {names}")
        edge = allowed[0]
        record = HistoryRecord(self._clock(), self.state, event, edge.target, note)
        self.state = edge.target
        self.history.append(record)
        for callback in self._on_enter.get(edge.target, ()):
            callback(record)
        return edge

    def force(self, state: str, note: str) -> None:
        """Синхронизация с реальностью (например, после чтения состояния устройства)."""
        if state not in self.definition.states:
            raise ValueError(f"{self.definition.name}: неизвестное состояние {state!r}")
        self.history.append(HistoryRecord(self._clock(), self.state, "<sync>", state, note))
        self.state = state
