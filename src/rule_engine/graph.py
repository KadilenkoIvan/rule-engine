"""Префиксный граф сценариев: правила с общим началом делят вершины."""

from dataclasses import dataclass, field
from typing import NamedTuple

from rule_engine.rules import RuleSet


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


def render(starts: dict[Start, Node], rules: RuleSet) -> str:
    """Задержку и порог показывает у рёбер, где они отличаются от `max_gap` и `min_score` корня."""
    lines = ["(начало)"]

    def label(edge: Start | Edge) -> str:
        notes = []
        if isinstance(edge, Edge) and edge.max_gap != rules.max_gap:
            notes.append(f"{edge.max_gap:g} с")
        if edge.min_score != rules.min_score:
            notes.append(f"≥ {edge.min_score:g}")
        return f"{edge.token} [{', '.join(notes)}]" if notes else edge.token

    def walk(edges: dict[Start, Node] | dict[Edge, Node], indent: str) -> None:
        for index, (edge, node) in enumerate(edges.items()):
            last = index == len(edges) - 1
            mark = f"  ⇒ {node.scenario}" if node.scenario is not None else ""
            lines.append(f"{indent}{'└─ ' if last else '├─ '}{label(edge)}{mark}")
            walk(node.children, indent + ("   " if last else "│  "))

    walk(starts, "")
    return "\n".join(lines)
