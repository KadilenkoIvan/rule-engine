"""Проход потока токенов по графу сценариев."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from rule_engine.graph import Node, build_graph
from rule_engine.rules import RuleSet


@dataclass(frozen=True)
class Event:
    token: str
    score: float  # уверенность детектора; в сопоставлении пока не участвует

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"уверенность {self.score} токена {self.token!r} вне [0, 1]")


class Step(NamedTuple):
    position: int  # сквозной номер события в потоке, между порциями не сбрасывается
    event: Event


@dataclass(frozen=True)
class Match:
    scenario: str
    steps: tuple[Step, ...]


class Engine:
    """Частичные совпадения хранятся между вызовами, поэтому поток можно подавать порциями.

    - Токен, не продолжающий частичное совпадение, пропускается: совпадение ждёт дальше.
    - Токен, продолжающий совпадение, переводит его в следующую вершину, а не копирует.
    - В вершине живёт не больше одного совпадения; из двух остаётся начавшееся позже.
      Поэтому активных совпадений не больше, чем вершин в графе.
    - Сценарий срабатывает при входе в его конечную вершину. Если из неё есть продолжения
      (более длинные сценарии с тем же началом), совпадение остаётся активным.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._vocabulary = rules.vocabulary
        self._root = build_graph(rules)
        self._active: dict[Node, tuple[Step, ...]] = {}
        self._position = 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    def feed(self, events: Iterable[Event]) -> list[Match]:
        matches: list[Match] = []
        for event in events:
            matches.extend(self.push(event))
        return matches

    def push(self, event: Event) -> list[Match]:
        if event.token not in self._vocabulary:
            raise ValueError(f"токена {event.token!r} нет в словаре")
        step = Step(self._position, event)
        self._position += 1

        matches: list[Match] = []
        active: dict[Node, tuple[Step, ...]] = {}

        def keep(node: Node, path: tuple[Step, ...]) -> None:
            current = active.get(node)
            if current is None or path[0].position > current[0].position:
                active[node] = path

        def enter(node: Node, path: tuple[Step, ...]) -> None:
            if node.scenario is not None:
                matches.append(Match(node.scenario, path))
            if node.children:
                keep(node, path)

        for node, path in self._active.items():
            child = node.children.get(event.token)
            if child is None:
                keep(node, path)
            else:
                enter(child, path + (step,))
        start = self._root.children.get(event.token)
        if start is not None:
            enter(start, (step,))

        self._active = active
        return matches
