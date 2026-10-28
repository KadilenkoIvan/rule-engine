"""Совпадения: события потока, частичные совпадения и то, что движок о них сообщает."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from rule_engine.domain.graph import Node
from rule_engine.domain.unordered import Unordered

if TYPE_CHECKING:
    from rule_engine.domain.front import Owner

# где стоит совпадение: вершина бора или (шаблон `any`, номера закрытых шагов)
Place = Node | tuple[Unordered, frozenset[int]]


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
    confidence: float  # общая уверенность: среднее геометрическое score шагов


class Partial(NamedTuple):
    """Частичное совпадение: его шаги и произведение их score."""

    steps: tuple[Step, ...]
    product: float

    @property
    def last_time(self) -> float:
        """Время последнего шага: от него считается срок продолжения."""
        return self.steps[-1].event.time

    @property
    def confidence(self) -> float:
        """Среднее геометрическое score шагов."""
        return self.product ** (1 / len(self.steps))

    def bound(self, length: int) -> float:
        """Уверенность в шаблоне длины `length`, если у оставшихся шагов score 1.0: выше не бывает."""
        return self.product ** (1 / length)

    def extend(self, step: Step) -> "Partial":
        return Partial(self.steps + (step,), self.product * step.event.score)

    def positions(self) -> tuple[int, ...]:
        return tuple(step.position for step in self.steps)


def dominates(a: Partial, b: Partial) -> bool:
    """`a` вытесняет `b` из вершины: последний шаг у `a` не раньше, а произведение не меньше.

    Тогда любое продолжение `b` есть и у `a`, со сроком не раньше и уверенностью не ниже. Если
    равны и последний шаг, и произведение, остаётся начавшееся позже, а при равном начале — с более
    поздними шагами: так исход не зависит от того, в каком порядке совпадения пришли в вершину.
    """
    if (a.steps[-1].position, a.product) == (b.steps[-1].position, b.product):
        return a.positions() >= b.positions()
    return a.steps[-1].position >= b.steps[-1].position and a.product >= b.product


class Waiting(NamedTuple):
    """Частичное совпадение в снимке движка: где оно стоит и до какого момента ждёт."""

    place: Place
    partial: Partial
    until: float  # секунды: срок продолжения


class Miss(NamedTuple):
    """Почему событие не продвинуло совпадение или не дало сработать сценарию.

    - `"score"`: score токена `got` ниже порога шага `need`. `place` — куда не пустило: вершина
      за ребром или состояние набора `any`.
    - `"late"`: токен подошёл бы, но пришёл в `got`, а ждали до `need`. `place` — как у `"score"`.
    - `"expired"`: совпадение снято: ждали до `need`, а токен в `got` ему не подходил. `place` —
      где оно стояло.
    - `"pruned"`: совпадение отсеяно: даже при score 1.0 у оставшихся шагов уверенность не выше
      `got`, а порог `need`. `place` — куда оно вошло.
    - `"confidence"`: шаблон сценария собран, но общая уверенность `got` ниже порога `need`.
    - `"repeat"`: шаблон собран, но его токен уже вошёл в выданное срабатывание сценария.
    - `"cooldown"`: шаблон собран, но сценарий молчит до `need`.

    `partial` — совпадение, которое не продвинулось; `None` — новое совпадение не началось.
    У трёх последних причин есть `scenario`, а `place` — `None`.
    """

    reason: str
    place: Place | None
    partial: Partial | None
    scenario: str | None
    need: float
    got: float


class Tick:
    """Одно событие в проходе: что оно завершило, куда продлило совпадения и что не продвинуло.

    Продлённые совпадения встают на места только после обхода (`entries`), чтобы событие не
    продолжило совпадение, которое само же и продлило.
    """

    __slots__ = ("step", "completed", "entries", "misses")

    def __init__(self, step: Step) -> None:
        self.step = step
        self.completed: dict[str, list[Partial]] = {}  # сценарий → совпадения, завершённые событием
        self.entries: list[tuple["Owner", Any, Partial]] = []  # (чьё место, какое, кто входит)
        self.misses: list[Miss] = []

    @property
    def event(self) -> Event:
        return self.step.event

    def complete(self, scenario: str, partials: Iterable[Partial]) -> None:
        self.completed.setdefault(scenario, []).extend(partials)

    def enter(self, owner: "Owner", key: Any, partial: Partial) -> None:
        self.entries.append((owner, key, partial))
