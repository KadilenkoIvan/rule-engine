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
    edge: TokenEdge  # ребро, которое ведёт в вершину
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


@dataclass(frozen=True)
class Frame:
    """Состояние после события."""

    event: EventView
    matches: tuple[MatchView, ...]
    waiting: tuple[WaitingView, ...]
