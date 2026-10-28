"""Префиксный граф сценариев: правила с общим началом делят вершины.

В граф входят сценарии с `order: "sequence"`; шаблоны `any` собирает unordered.py.
"""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

from rule_engine.domain.rules import PatternStep, RuleSet, Scenario

# ранний отсев сравнивает произведение с `floor`, а рядом с ним — прежней формулой; доля от `floor`
FLOOR_SLACK = 1e-9


class Start(NamedTuple):
    """Ребро из корня. У первого шага задержки нет: новое совпадение её не проверяет."""

    token: str
    min_score: float  # порог score токена


class Edge(NamedTuple):
    token: str
    max_gap: float  # секунды от предыдущего шага
    min_score: float  # порог score токена


# созданное ребро: (откуда, ребро, куда); у рёбер из корня «откуда» — None
Created = tuple["Node | None", Start | Edge, "Node"]


@dataclass(eq=False)
class Node:
    """Сравнивается по identity: вершина служит ключом для активных совпадений."""

    # шаги с одним токеном, но разной задержкой или порогом — разные рёбра: иначе одному совпадению
    # в общей вершине пришлось бы ждать по срокам и порогам разных сценариев
    children: dict[Edge, "Node"] = field(default_factory=dict)
    scenario: str | None = None  # сценарий, один из шаблонов которого завершается в этой вершине
    # (длина шаблона, порог общей уверенности его сценария) для шаблонов, которые кончаются ниже
    # вершины: по ним движок снимает совпадение, которое не дотянет ни до одного порога
    ahead: set[tuple[int, float]] = field(default_factory=set)
    max_gap: float = 0.0  # наибольшая задержка рёбер: позже неё совпадению некуда продолжиться
    # `product ** (1 / L) >= threshold` для какого-то шаблона из `ahead` равносильно
    # `product >= threshold ** L`, поэтому для раннего отсева хватает минимума по `ahead`
    floor: float = math.inf

    def expect(self, length: int, threshold: float) -> None:
        """Запоминает шаблон длины `length` с порогом `threshold`, который кончается ниже."""
        self.ahead.add((length, threshold))
        self.floor = min(self.floor, threshold ** length)

    def promising(self, product: float) -> bool:
        """Ранний отсев: дотянет ли совпадение с таким произведением хоть до одного порога ниже.

        У вершины без продолжений `ahead` пуст: совпадение в ней не остаётся.
        """
        # вдали от `floor` хватает сравнения с ним; рядом — та же формула, что в отчёте об
        # отсеве, чтобы исход на самой границе не решало округление
        if product > self.floor * (1 + FLOOR_SLACK):
            return True
        if product < self.floor * (1 - FLOOR_SLACK):
            return False
        return any(product ** (1 / length) >= threshold for length, threshold in self.ahead)

    def nearest(self, product: float) -> tuple[int, float]:
        """Шаблон ниже, до порога которого совпадению ближе всего: (длина, порог)."""
        return max(self.ahead, key=lambda item: product ** (1 / item[0]) - item[1])


def build_graph(rules: RuleSet) -> dict[Start, Node]:
    """Рёбра из корня: токен и порог первого шага → вершина.

    Дальше ребро — токен, действующие задержка и порог шага: общее начало у сценариев общее,
    только если настройки на нём совпадают.
    """
    starts: dict[Start, Node] = {}
    for scenario in rules.scenarios:
        insert_scenario(starts, rules, scenario)
    return starts


def insert_scenario(starts: dict[Start, Node], rules: RuleSet, scenario: Scenario) -> list[Created]:
    """Дописывает пути проверенного сценария, сохраняя общие вершины; возвращает созданные рёбра.

    Конфликты полных шаблонов проверяются до изменения бора, по прежним правилам.
    """
    if scenario.order != "sequence":
        return []
    paths = [pattern_path(rules, scenario, pattern) for pattern in scenario.patterns]
    for start, edges in paths:
        node = find(starts, start, edges)
        if node is not None and node.scenario is not None:
            tokens = [start.token, *(edge.token for edge in edges)]
            raise ValueError(
                f"сценарии {node.scenario!r} и {scenario.name!r} задают одну и ту же "
                f"последовательность токенов {tokens} с теми же задержками и порогами"
            )

    created: list[Created] = []
    threshold = rules.confidence_threshold(scenario)
    for start, edges in paths:
        nodes = grow(starts, start, edges, created)
        nodes[-1].scenario = scenario.name
        for above in nodes[:-1]:
            above.expect(len(nodes), threshold)
    return created


def pattern_path(
    rules: RuleSet, scenario: Scenario, pattern: tuple[PatternStep, ...]
) -> tuple[Start, tuple[Edge, ...]]:
    """Шаблон как путь в боре: ребро из корня и следующие рёбра с действующими настройками."""
    first, *rest = pattern
    start = Start(first.token, rules.threshold(scenario, first))
    edges = tuple(
        Edge(step.token, rules.gap(scenario, step), rules.threshold(scenario, step))
        for step in rest
    )
    return start, edges


def find(starts: dict[Start, Node], start: Start, edges: tuple[Edge, ...]) -> Node | None:
    """Вершина в конце пути, если весь путь уже есть в боре."""
    node = starts.get(start)
    for edge in edges:
        if node is None:
            return None
        node = node.children.get(edge)
    return node


def grow(
    starts: dict[Start, Node], start: Start, edges: tuple[Edge, ...], created: list[Created]
) -> list[Node]:
    """Дописывает недостающие вершины пути в бор и в `created`; возвращает вершины пути."""
    if start not in starts:
        starts[start] = Node()
        created.append((None, start, starts[start]))
    nodes = [starts[start]]
    for edge in edges:
        node = nodes[-1]
        if edge not in node.children:
            node.children[edge] = Node()
            node.max_gap = max(node.max_gap, edge.max_gap)
            created.append((node, edge, node.children[edge]))
        nodes.append(node.children[edge])
    return nodes


def graph_size(starts: dict[Start, Node]) -> int:
    """Число вершин без корня."""

    def size(node: Node) -> int:
        return 1 + sum(size(child) for child in node.children.values())

    return sum(size(node) for node in starts.values())
