"""View models: plain data that each interface shows its own way.

No domain objects here: the CLI prints the graph as a tree, the web draws it as SVG, and
neither needs to know engine internals.
"""

from dataclasses import dataclass
from typing import NamedTuple


class TokenEdge(NamedTuple):
    """Token edge; gap and threshold are None when equal to the root rules'."""

    token: str
    max_gap: float | None  # у ребра из корня всегда None: новое совпадение задержку не проверяет
    min_score: float | None


class SilenceEdge(NamedTuple):
    """Silence edge; the threshold is None when equal to the root rules'."""

    tokens: tuple[str, ...]  # запрещённые токены, по алфавиту
    duration: float  # секунды вперёд
    before: float  # секунды назад; 0 — назад не смотрит
    min_score: float | None


@dataclass(frozen=True)
class NodeView:
    """Node of the logical trie with the edge into it."""

    id: int  # номер вершины; по нему снимок совпадений ссылается на граф
    edge: TokenEdge | SilenceEdge  # ребро, которое ведёт в вершину
    scenario: str | None  # сценарий, чей шаблон кончается в вершине
    children: tuple["NodeView", ...]  # сначала за рёбрами-токенами, потом за рёбрами-тишинами


class SlotView(NamedTuple):
    """Step of an `any` pattern."""

    token: str
    min_score: float | None  # None — как в корне правил


class SetView(NamedTuple):
    """`any` pattern."""

    id: int  # номер шаблона; по нему снимок совпадений ссылается на граф
    scenario: str
    slots: tuple[SlotView, ...]
    max_gap: float | None  # None — как в корне правил


@dataclass(frozen=True)
class GraphView:
    """Graph of the rules: the trie, the `any` patterns and the vocabulary."""

    roots: tuple[NodeView, ...]  # вершины за рёбрами из корня
    sets: tuple[SetView, ...]
    vocabulary: tuple[str, ...]  # по алфавиту


class EventView(NamedTuple):
    """Stream event."""

    position: int  # сквозной номер события в потоке
    token: str
    score: float
    time: float


class MatchView(NamedTuple):
    """Fired scenario."""

    scenario: str
    steps: tuple[EventView, ...]
    confidence: float
    released: float | None  # режим поправок: срок, после которого выдано; None — выдано сразу


class HeldView(NamedTuple):
    """Fired scenario whose release waits (correction mode)."""

    scenario: str
    steps: tuple[EventView, ...]
    confidence: float
    until: float  # выдаётся, когда время станет больше
    frozen: bool  # ждёт решения маркера, токен которого стоит шагом живого совпадения


class WaitingView(NamedTuple):
    """Partial match waiting for a continuation."""

    # "node" — в вершине бора, "set" — в наборе `any`, "silence" — в окне тишины
    kind: str
    place: int  # id вершины или набора; у окна — id вершины за ребром-тишиной
    filled: frozenset[int]  # закрытые шаги набора; у вершины пусто
    steps: tuple[EventView, ...]
    confidence: float
    until: float  # срок продолжения


class MissView(NamedTuple):
    """Why an event did not advance a match or let a scenario fire.

    `reason`, `need` and `got` mean the same as in `Miss` of the engine.
    """

    reason: str
    kind: str | None  # "node", "set" или "silence"; None — отказ сценария
    place: int | None  # id вершины или набора, у окна — вершины за ним; None — отказ сценария
    filled: frozenset[int]  # закрытые шаги набора; иначе пусто
    steps: tuple[EventView, ...]  # шаги совпадения; пусто — новое совпадение не началось
    scenario: str | None  # у отказов сценария
    need: float
    got: float


@dataclass(frozen=True)
class Frame:
    """State after an event, or after a deadline passed without an event."""

    time: float  # секунды: время события или конец окна
    event: EventView | None  # None — кадр по таймеру
    matches: tuple[MatchView, ...]
    waiting: tuple[WaitingView, ...]
    held: tuple[HeldView, ...]  # ждут выдачи
    misses: tuple[MissView, ...]  # что событие не продвинуло и почему
