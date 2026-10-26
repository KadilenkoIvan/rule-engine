"""Префиксный граф сценариев: правила с общим началом делят вершины.

В граф входят сценарии с `order: "sequence"`; шаблоны `any` собирает unordered.py.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

from rule_engine.domain.rules import RuleSet, Scenario


class Start(NamedTuple):
    """Ребро из корня. У первого шага задержки нет: новое совпадение её не проверяет."""

    token: str
    min_score: float  # порог score токена


class Edge(NamedTuple):
    token: str
    max_gap: float  # секунды от предыдущего шага
    min_score: float  # порог score токена


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


def build_graph(rules: RuleSet) -> dict[Start, Node]:
    """Рёбра из корня: токен и порог первого шага → вершина.

    Дальше ребро — токен, действующие задержка и порог шага: общее начало у сценариев общее,
    только если настройки на нём совпадают.
    """
    starts: dict[Start, Node] = {}
    for scenario in rules.scenarios:
        insert_scenario(starts, rules, scenario)
    return starts


def insert_scenario(starts: dict[Start, Node], rules: RuleSet, scenario: Scenario) -> None:
    """Дописывает пути проверенного сценария, сохраняя общие вершины.

    Конфликты полных шаблонов проверяются до изменения бора, по прежним правилам.
    """
    if scenario.order != "sequence":
        return
    paths: list[tuple[Start, tuple[Edge, ...]]] = []
    for first, *rest in scenario.patterns:
        start = Start(first.token, rules.threshold(scenario, first))
        edges = tuple(
            Edge(step.token, rules.gap(scenario, step), rules.threshold(scenario, step))
            for step in rest
        )
        paths.append((start, edges))
        node = starts.get(start)
        for edge in edges:
            if node is None:
                break
            node = node.children.get(edge)
        if node is not None and node.scenario is not None:
            tokens = [start.token, *(edge.token for edge in edges)]
            raise ValueError(
                f"сценарии {node.scenario!r} и {scenario.name!r} задают одну и ту же "
                f"последовательность токенов {tokens} с теми же задержками и порогами"
            )

    threshold = rules.confidence_threshold(scenario)
    for start, edges in paths:
        if start not in starts:
            starts[start] = Node()
        node = starts[start]
        nodes = [node]
        for edge in edges:
            if edge not in node.children:
                node.children[edge] = Node()
            node = node.children[edge]
            nodes.append(node)
        node.scenario = scenario.name
        for above in nodes[:-1]:
            above.ahead.add((len(nodes), threshold))


def graph_size(starts: dict[Start, Node]) -> int:
    """Число вершин без корня."""

    def size(node: Node) -> int:
        return 1 + sum(size(child) for child in node.children.values())

    return sum(size(node) for node in starts.values())

