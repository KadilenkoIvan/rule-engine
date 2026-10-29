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


class SilenceEdge(NamedTuple):
    """Ребро-тишина. Порог — `None`, если совпадает с корнем правил."""

    tokens: tuple[str, ...]  # запрещённые токены, по алфавиту
    duration: float  # секунды вперёд
    before: float  # секунды назад; 0 — назад не смотрит
    min_score: float | None


@dataclass(frozen=True)
class NodeView:
    id: int  # номер вершины; по нему снимок совпадений ссылается на граф
    edge: TokenEdge | SilenceEdge  # ребро, которое ведёт в вершину
    scenario: str | None  # сценарий, чей шаблон кончается в вершине
    children: tuple["NodeView", ...]  # сначала за рёбрами-токенами, потом за рёбрами-тишинами


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
    released: float | None  # режим поправок: срок, после которого выдано; None — выдано сразу


class HeldView(NamedTuple):
    """Сработавший сценарий, чья выдача ждёт срока (режим поправок)."""

    scenario: str
    steps: tuple[EventView, ...]
    confidence: float
    until: float  # выдаётся, когда время станет больше
    frozen: bool  # ждёт решения маркера, токен которого стоит шагом живого совпадения


class WaitingView(NamedTuple):
    """Частичное совпадение, которое ждёт продолжения."""

    # "node" — в вершине бора, "set" — в наборе `any`, "silence" — в окне тишины
    kind: str
    place: int  # id вершины или набора; у окна — id вершины за ребром-тишиной
    filled: frozenset[int]  # закрытые шаги набора; у вершины пусто
    steps: tuple[EventView, ...]
    confidence: float
    until: float  # срок продолжения


class MissView(NamedTuple):
    """Почему событие не продвинуло совпадение или не дало сработать сценарию.

    `reason` и смысл `need` и `got` — как у `Miss` в движке: `"score"`, `"late"`, `"expired"`,
    `"pruned"`, `"silence"`, `"confidence"`, `"repeat"`, `"cooldown"`, `"corrected"`,
    `"negated"`.
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
    """Состояние после события или после конца окна тишины, когда время прошло без события."""

    time: float  # секунды: время события или конец окна
    event: EventView | None  # None — кадр по таймеру
    matches: tuple[MatchView, ...]
    waiting: tuple[WaitingView, ...]
    held: tuple[HeldView, ...]  # ждут выдачи
    misses: tuple[MissView, ...]  # что событие не продвинуло и почему
