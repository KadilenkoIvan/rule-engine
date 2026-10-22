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
    score: float  # уверенность детектора: шаг принимает токен, только если она не ниже порога
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
    - У ребра есть задержка: совпадение продолжается по нему, только если от последнего шага
      совпадения прошло не больше неё. Посторонние токены этот срок не продлевают. Когда сроки
      всех рёбер из вершины истекли, совпадение снимается. Новое совпадение от корня задержку
      не проверяет.
    - У ребра есть порог: токен проходит по нему, только если его score не ниже порога. Иначе
      для этого ребра токена нет, и совпадение ждёт дальше. Новое совпадение тоже начинается,
      только если score токена не ниже порога ребра из корня.
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
    - Время событий не убывает: событие раньше предыдущего — ошибка.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._vocabulary = rules.vocabulary
        self._cooldowns = {scenario.name: scenario.cooldown for scenario in rules.scenarios}
        self._starts = build_graph(rules)
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
        if self._last_time is not None and event.time < self._last_time:
            raise ValueError(
                f"время {event.time} токена {event.token!r} раньше предыдущего {self._last_time}"
            )
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
            # срок считается от последнего шага совпадения, а не от последнего токена потока
            elapsed = event.time - path[-1].event.time
            if all(elapsed > edge.max_gap for edge in node.children):
                continue  # сроки всех рёбер из вершины истекли: совпадение снято
            keep(node, path)  # копия остаётся в вершине и ждёт другого продолжения
            for edge, child in node.children.items():
                if (
                    edge.token == event.token
                    and elapsed <= edge.max_gap
                    and event.score >= edge.min_score
                ):
                    enter(child, path + (step,))
        for start, child in self._starts.items():
            if start.token == event.token and event.score >= start.min_score:
                enter(child, (step,))  # новое совпадение задержку не проверяет

        self._active = active
        # новые срабатывания состоят из позиций не раньше начала самого старого активного
        # совпадения, более ранние можно забыть
        oldest = min((path[0].position for path in active.values()), default=self._position)
        for taken in self._taken.values():
            taken.difference_update([position for position in taken if position < oldest])
        return matches
