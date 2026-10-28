"""View-модели: чистые данные, которые интерфейсы показывают каждый по-своему.

Объектов домена здесь нет: CLI печатает граф деревом, веб рисует его в SVG, и обоим не нужно
знать, как устроены вершины и совпадения внутри движка.
"""

from dataclasses import dataclass
from typing import NamedTuple


class TokenEdge(NamedTuple):
    """Ребро-токен. Задержка и порог — `None`, если совпадают с корнем правил."""

    token: str
    max_gap: float | None  # у ребра из корня всегда None: новое совпадение задержку не проверяет
    min_score: float | None


@dataclass(frozen=True)
class NodeView:
    id: int  # номер вершины; по нему снимок совпадений ссылается на граф
    segment: tuple[TokenEdge, ...]  # шаги сжатого ребра, ведущего в вершину
    scenario: str | None  # сценарий, чей шаблон кончается в вершине
    children: tuple["NodeView", ...]


class SlotView(NamedTuple):
    token: str
    min_score: float | None  # None — как в корне правил


class SetView(NamedTuple):
    """Шаблон `any`."""

    id: int  # номер шаблона; по нему снимок совпадений ссылается на граф
    scenario: str
    slots: tuple[SlotView, ...]
    max_gap: float | None  # None — как в корне правил


@dataclass(frozen=True)
class GraphView:
    roots: tuple[NodeView, ...]  # вершины за рёбрами из корня
    sets: tuple[SetView, ...]
    vocabulary: tuple[str, ...]  # по алфавиту


class EventView(NamedTuple):
    position: int  # сквозной номер события в потоке
    token: str
    score: float
    time: float


class MatchView(NamedTuple):
    scenario: str
    steps: tuple[EventView, ...]
    confidence: float


class WaitingView(NamedTuple):
    """Частичное совпадение, которое ждёт продолжения."""

    kind: str  # "node" — в вершине бора, "set" — в наборе `any`
    place: int  # id вершины или набора
    filled: frozenset[int]  # закрытые шаги набора; у вершины пусто
    steps: tuple[EventView, ...]
    confidence: float
    until: float  # срок продолжения
    offset: int = 0  # число принятых шагов сжатого ребра; у any — 0


class MissView(NamedTuple):
    """Почему событие не продвинуло совпадение или не дало сработать сценарию.

    `reason` и смысл `need` и `got` — как у `Miss` в движке: `"score"`, `"late"`, `"expired"`,
    `"pruned"`, `"confidence"`, `"repeat"`, `"cooldown"`.
    """

    reason: str
    kind: str | None  # "node" или "set"; None — отказ сценария
    place: int | None  # id вершины или набора; None — отказ сценария
    filled: frozenset[int]  # закрытые шаги набора; иначе пусто
    steps: tuple[EventView, ...]  # шаги совпадения; пусто — новое совпадение не началось
    scenario: str | None  # у отказов сценария
    need: float
    got: float
    offset: int = 0  # позиция отказа внутри сжатого ребра


@dataclass(frozen=True)
class Frame:
    """Состояние после события."""

    event: EventView
    matches: tuple[MatchView, ...]
    waiting: tuple[WaitingView, ...]
    misses: tuple[MissView, ...]  # что событие не продвинуло и почему
