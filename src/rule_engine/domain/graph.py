"""Префиксный граф сценариев: правила с общим началом делят вершины.

В граф входят сценарии с `order: "sequence"`; шаблоны `any` собирает unordered.py.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

from rule_engine.domain.rules import RuleSet


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
        if scenario.order != "sequence":
            continue
        for pattern in scenario.patterns:
            first, *rest = pattern
            start = Start(first.token, rules.threshold(scenario, first))
            if start not in starts:
                starts[start] = Node()
            node = starts[start]
            nodes = [node]
            for step in rest:
                edge = Edge(step.token, rules.gap(scenario, step), rules.threshold(scenario, step))
                if edge not in node.children:
                    node.children[edge] = Node()
                node = node.children[edge]
                nodes.append(node)
            if node.scenario is not None:
                raise ValueError(
                    f"сценарии {node.scenario!r} и {scenario.name!r} задают одну и ту же "
                    f"последовательность токенов {[step.token for step in pattern]} с теми же "
                    "задержками и порогами"
                )
            node.scenario = scenario.name
            for above in nodes[:-1]:
                above.ahead.add((len(pattern), rules.confidence_threshold(scenario)))
    return starts


def graph_size(starts: dict[Start, Node]) -> int:
    """Число вершин без корня."""

    def size(node: Node) -> int:
        return 1 + sum(size(child) for child in node.children.values())

    return sum(size(node) for node in starts.values())

