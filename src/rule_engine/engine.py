"""Проход потока токенов по графу сценариев."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from rule_engine.graph import Node, build_graph
from rule_engine.rules import RuleSet


@dataclass(frozen=True)
class Event:
    token: str
    score: float  # уверенность детектора; в сопоставлении пока не участвует
    time: float  # секунды: когда токен сказан, по одной шкале для всего потока (ставит детектор)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"уверенность {self.score} токена {self.token!r} вне [0, 1]")
        if not math.isfinite(self.time):
            raise ValueError(f"время {self.time} токена {self.token!r} не конечное число")


class Step(NamedTuple):
    position: int  # сквозной номер события в потоке, между порциями не сбрасывается
    event: Event


@dataclass(frozen=True)
class Match:
    scenario: str
    steps: tuple[Step, ...]


def freshness(path: tuple[Step, ...]) -> tuple[int, int]:
    """Позиции первого и последнего шага: из двух совпадений в вершине остаётся то, у кого больше."""
    return path[0].position, path[-1].position


class Engine:
    """Частичные совпадения хранятся между вызовами, поэтому поток можно подавать порциями.

    - Токен, не продолжающий частичное совпадение, пропускается: совпадение ждёт дальше.
    - Совпадение, продолжившееся по ребру, остаётся и в старой вершине: копия ждёт другого
      продолжения. Поэтому лишний токен со своим ребром не уводит совпадение в чужую ветку.
    - В вершине живёт не больше одного совпадения; из двух остаётся начавшееся позже, а при
      равном начале — то, у которого последний шаг позже. Поэтому активных совпадений не
      больше, чем вершин в графе.
    - Сценарий срабатывает при входе в вершину, где заканчивается один из его шаблонов. Если
      из неё есть продолжения (более длинные шаблоны с тем же началом), совпадение остаётся
      активным.
    - Срабатывание не выдаётся, если у него есть общий токен с выданным срабатыванием того же
      сценария (токены сравниваются по позиции в потоке) или если с выданного срабатывания
      этого сценария прошло меньше его `cooldown`. Невыданное срабатывание не занимает токены
      и не продлевает cooldown, а совпадение живёт дальше.
    - Пауза между соседними токенами больше `max_gap` начинает новую последовательность:
      все частичные совпадения снимаются, токен ищет сценарий с начала.
    - Время событий не убывает: событие раньше предыдущего — ошибка.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._vocabulary = rules.vocabulary
        self._max_gap = rules.max_gap
        self._cooldowns = {scenario.name: scenario.cooldown for scenario in rules.scenarios}
        self._root = build_graph(rules)
        self._active: dict[Node, tuple[Step, ...]] = {}
        self._taken: dict[str, set[int]] = {}  # сценарий → позиции токенов его выданных срабатываний
        self._fired_at: dict[str, float] = {}  # сценарий → время его последнего выданного срабатывания
        self._position = 0
        self._last_time: float | None = None  # None, пока не пришло ни одного события

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
        if self._last_time is not None:
            if event.time < self._last_time:
                raise ValueError(
                    f"время {event.time} токена {event.token!r} раньше предыдущего {self._last_time}"
                )
            if event.time - self._last_time > self._max_gap:
                self._active = {}
        self._last_time = event.time
        step = Step(self._position, event)
        self._position += 1

        matches: list[Match] = []
        active: dict[Node, tuple[Step, ...]] = {}

        def keep(node: Node, path: tuple[Step, ...]) -> None:
            current = active.get(node)
            if current is None or freshness(path) > freshness(current):
                active[node] = path

        def fire(scenario: str, path: tuple[Step, ...]) -> None:
            positions = {position for position, _ in path}
            taken = self._taken.setdefault(scenario, set())
            if not taken.isdisjoint(positions):
                return  # повтор: токен уже вошёл в выданное срабатывание этого сценария
            fired_at = self._fired_at.get(scenario)
            if fired_at is not None and event.time - fired_at < self._cooldowns[scenario]:
                return
            taken.update(positions)
            self._fired_at[scenario] = event.time
            matches.append(Match(scenario, path))

        def enter(node: Node, path: tuple[Step, ...]) -> None:
            if node.scenario is not None:
                fire(node.scenario, path)
            if node.children:
                keep(node, path)

        for node, path in self._active.items():
            keep(node, path)  # копия остаётся в вершине и ждёт другого продолжения
            child = node.children.get(event.token)
            if child is not None:
                enter(child, path + (step,))
        start = self._root.children.get(event.token)
        if start is not None:
            enter(start, (step,))

        self._active = active
        # новые срабатывания состоят из позиций не раньше начала самого старого активного
        # совпадения, более ранние можно забыть
        oldest = min((path[0].position for path in active.values()), default=self._position)
        for taken in self._taken.values():
            taken.difference_update([position for position in taken if position < oldest])
        return matches
