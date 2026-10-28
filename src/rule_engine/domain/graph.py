"""Radix trie: токеновые цепочки сжаты между развилками, завершениями и тишинами.

В граф входят сценарии с `order: "sequence"`; шаблоны `any` собирает unordered.py.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

from rule_engine.domain.rules import PatternStep, RuleSet, Scenario, SilenceStep


class Start(NamedTuple):
    """Ребро из корня. У первого шага задержки нет: новое совпадение её не проверяет."""

    token: str
    min_score: float  # порог score токена


class Edge(NamedTuple):
    token: str
    max_gap: float  # секунды от предыдущего шага
    min_score: float  # порог score токена


class Silence(NamedTuple):
    """Ребро-тишина: окно `duration` секунд без токенов `tokens` со score не ниже порога."""

    tokens: tuple[str, ...]  # по алфавиту: ребро не зависит от порядка в JSON
    duration: float  # секунды вперёд от предыдущего шага
    before: float  # секунды назад от предыдущего шага; 0 — назад не смотрим
    min_score: float  # порог score запрещённых токенов


class Targets:
    """Завершения шаблонов ниже позиции: (длина шаблона, порог общей уверенности сценария).

    По ним движок снимает совпадение, которое не дотянет ни до одного порога (ранний отсев).
    Неравенство `product ** (1 / L) >= порог` для завершений одной длины L выполнено хоть для
    одного, только если выполнено для наименьшего порога, поэтому отсев проверяет один порог на
    длину той же формулой.
    """

    __slots__ = ("items", "lowest")

    def __init__(self) -> None:
        self.items: set[tuple[int, float]] = set()
        self.lowest: dict[int, float] = {}  # длина шаблона → наименьший порог среди шаблонов этой длины

    def add(self, target: tuple[int, float]) -> None:
        self.items.add(target)
        length, threshold = target
        if threshold < self.lowest.get(length, math.inf):
            self.lowest[length] = threshold

    def update(self, other: "Targets") -> None:
        for target in other.items:
            self.add(target)

    def __iter__(self) -> Iterator[tuple[int, float]]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def reaches(self, product: float) -> bool:
        """Дотянет ли совпадение с этим произведением score хоть до одного порога."""
        return any(product ** (1 / length) >= threshold for length, threshold in self.lowest.items())


@dataclass(eq=False, slots=True)
class Node:
    """Конец сжатого ребра; сравнивается по identity, курсоры ссылаются на этот объект."""

    segment: tuple[Start | Edge | Silence, ...]
    # Все достижимые завершения, включая эту вершину: для позиций внутри segment.
    through: Targets = field(default_factory=Targets)
    # шаги с одним токеном, но разной задержкой или порогом — разные рёбра: иначе одному совпадению
    # в общей вершине пришлось бы ждать по срокам и порогам разных сценариев
    children: dict[Edge, "Node"] = field(default_factory=dict)
    # рёбра-тишины отдельно: их срок ведёт колесо таймеров, а не обход вершины на событии
    silences: dict[Silence, "Node"] = field(default_factory=dict)
    scenario: str | None = None  # сценарий, один из шаблонов которого завершается в этой вершине
    # (длина шаблона, порог общей уверенности его сценария) для шаблонов, которые кончаются ниже
    # вершины: по ним движок снимает совпадение, которое не дотянет ни до одного порога. Длина —
    # без шагов-тишин: в общую уверенность они не входят
    ahead: Targets = field(default_factory=Targets)
    max_gap: float = 0.0  # наибольшая задержка `children`: дольше в конце сегмента ждать нечего


class Cursor(NamedTuple):
    """Позиция после offset шагов сегмента; тишина всегда образует отдельный сегмент."""

    node: Node
    offset: int

    @property
    def scenario(self) -> str | None:
        return self.node.scenario if self.offset == len(self.node.segment) else None

    @property
    def ahead(self) -> Targets:
        return self.node.through if self.offset < len(self.node.segment) else self.node.ahead

    @property
    def max_gap(self) -> float:
        """Наибольшая задержка переходов: позже неё совпадению отсюда продолжиться некуда."""
        if self.offset < len(self.node.segment):
            return self.node.segment[self.offset].max_gap
        return self.node.max_gap

    def transitions(self) -> Iterator[tuple[Edge, "Cursor"]]:
        if self.offset < len(self.node.segment):
            yield self.node.segment[self.offset], Cursor(self.node, self.offset + 1)
        else:
            for edge, child in self.node.children.items():
                yield edge, Cursor(child, 1)

    def follow(self, edge: Edge) -> "Cursor":
        """Позиция за переходом `edge` отсюда."""
        if self.offset < len(self.node.segment):
            return Cursor(self.node, self.offset + 1)
        return Cursor(self.node.children[edge], 1)

    def silences(self) -> Iterator[Silence]:
        if self.offset == len(self.node.segment):
            yield from self.node.silences


def build_graph(rules: RuleSet) -> dict[Start, Node]:
    """Рёбра из корня: токен и порог первого шага → вершина.

    Дальше ребро — токен, действующие задержка и порог шага: общее начало у сценариев общее,
    только если настройки на нём совпадают.
    """
    starts: dict[Start, Node] = {}
    for scenario in rules.scenarios:
        insert_scenario(starts, rules, scenario)
    return starts


class Changes(NamedTuple):
    """Что вставка изменила в боре: по этому движок переносит и дописывает свои индексы."""

    splits: list[tuple[Node, Node, int]]  # (вершина, теперь суффикс; её префикс; длина префикса)
    starts: list[Start]  # новые рёбра из корня
    links: list[tuple[Node, Edge]]  # новые токеновые рёбра из конца сегмента вершины


def attach(node: Node, edge: Edge | Silence, child: Node) -> None:
    """Ребро из конца сегмента `node`; задержка конца сегмента — наибольшая у его рёбер."""
    if isinstance(edge, Silence):
        node.silences[edge] = child
    else:
        node.children[edge] = child
        node.max_gap = max(node.max_gap, edge.max_gap)


def insert_scenario(starts: dict[Start, Node], rules: RuleSet, scenario: Scenario) -> Changes:
    """Дописывает пути и возвращает изменения бора для переноса активных курсоров.

    Конфликты полных шаблонов проверяются до изменения бора, по прежним правилам.
    Старый Node остаётся концом суффикса: его рёбра-тишины и открытые Window стабильны.
    """
    changes = Changes([], [], [])
    if scenario.order != "sequence":
        return changes
    paths: list[tuple[Start | Edge | Silence, ...]] = []
    for first, *rest in scenario.patterns:
        start = Start(first.token, rules.threshold(scenario, first))
        path = (start, *(edge_of(rules, scenario, step) for step in rest))
        paths.append(path)
        node = starts.get(start)
        remaining = path
        while node is not None:
            if remaining[: len(node.segment)] != node.segment:
                break
            remaining = remaining[len(node.segment) :]
            if not remaining:
                if node.scenario is not None:
                    tokens = list(map(edge_label, path))
                    raise ValueError(
                        f"сценарии {node.scenario!r} и {scenario.name!r} задают одну и ту же "
                        f"последовательность токенов {tokens} с теми же задержками и порогами"
                    )
                break
            node = branches(node, remaining[0]).get(remaining[0])

    threshold = rules.confidence_threshold(scenario)
    for path in paths:
        remaining = path
        parent: Node | None = None
        ancestors = []
        target = (sum(not isinstance(edge, Silence) for edge in path), threshold)
        while True:
            key = remaining[0]
            children = starts if parent is None else branches(parent, key)
            node = children.get(key)
            if node is None:
                # Тишина не сливается ни с предыдущими, ни со следующими токенами.
                length = 1
                if not isinstance(key, Silence):
                    while length < len(remaining) and not isinstance(remaining[length], Silence):
                        length += 1
                node = Node(remaining[:length])
                if parent is None:
                    starts[key] = node
                    changes.starts.append(key)
                else:
                    attach(parent, key, node)
                    if isinstance(key, Edge):
                        changes.links.append((parent, key))
            else:
                common = 0
                for old, new in zip(node.segment, remaining):
                    if old != new:
                        break
                    common += 1
                if common < len(node.segment):
                    prefix = Node(node.segment[:common])
                    prefix.ahead.update(node.through)
                    prefix.through.update(node.through)
                    node.segment = node.segment[common:]
                    attach(prefix, node.segment[0], node)
                    children[key] = prefix  # ребро то же, поэтому задержка родителя прежняя
                    changes.splits.append((node, prefix, common))
                    node = prefix
            remaining = remaining[len(node.segment) :]
            if not remaining:
                break
            ancestors.append(node)
            parent = node
        node.scenario = scenario.name
        node.through.add(target)
        for above in ancestors:
            above.ahead.add(target)
            above.through.add(target)
    return changes


def edge_of(rules: RuleSet, scenario: Scenario, step: PatternStep | SilenceStep) -> Edge | Silence:
    """Ребро шага с действующими задержкой и порогом."""
    if isinstance(step, SilenceStep):
        return Silence(
            tuple(sorted(step.absent)), step.duration, step.before, rules.threshold(scenario, step)
        )
    return Edge(step.token, rules.gap(scenario, step), rules.threshold(scenario, step))


def branches(node: Node, edge: Start | Edge | Silence) -> dict:
    """Рёбра вершины того же вида, что и `edge`."""
    return node.silences if isinstance(edge, Silence) else node.children


def edge_label(edge: Start | Edge | Silence) -> str:
    if isinstance(edge, Silence):
        before = f", назад {edge.before:g} с" if edge.before else ""
        return f"тишина {list(edge.tokens)} {edge.duration:g} с{before}"
    return edge.token


def longest_before(starts: dict[Start, Node]) -> float:
    """Самое длинное окно тишины назад в боре; 0, если назад не смотрит ни одна тишина."""

    def walk(node: Node) -> float:
        return max(
            (
                max(edge.before, walk(child)) if isinstance(edge, Silence) else walk(child)
                for edge, child in (*node.children.items(), *node.silences.items())
            ),
            default=0.0,
        )

    return max((walk(node) for node in starts.values()), default=0.0)


def graph_size(starts: dict[Start, Node]) -> int:
    """Число физических вершин radix trie без корня."""

    def size(node: Node) -> int:
        children = (*node.children.values(), *node.silences.values())
        return 1 + sum(size(child) for child in children)

    return sum(size(node) for node in starts.values())
