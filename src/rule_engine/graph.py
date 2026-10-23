"""Префиксный граф сценариев: правила с общим началом делят вершины."""

from dataclasses import dataclass, field
from typing import NamedTuple

from rule_engine.rules import RuleSet


class Edge(NamedTuple):
    token: str
    max_gap: float  # секунды от предыдущего шага


@dataclass(eq=False)
class Node:
    """Сравнивается по identity: вершина служит ключом для активных совпадений."""

    # шаги с одним токеном, но разной задержкой — разные рёбра: иначе одному совпадению в общей
    # вершине пришлось бы ждать по срокам разных сценариев
    children: dict[Edge, "Node"] = field(default_factory=dict)
    scenario: str | None = None  # сценарий, один из шаблонов которого завершается в этой вершине


def build_graph(rules: RuleSet) -> dict[str, Node]:
    """Рёбра из корня: токен первого шага → вершина.

    У первого шага задержки нет, новое совпадение её не проверяет, поэтому ребро из корня задаёт
    только токен. Дальше ребро — токен и действующая задержка шага: общее начало у сценариев
    общее, только если задержки на нём совпадают.
    """
    starts: dict[str, Node] = {}
    for scenario in rules.scenarios:
        for pattern in scenario.patterns:
            first, *rest = pattern
            if first.token not in starts:
                starts[first.token] = Node()
            node = starts[first.token]
            for step in rest:
                edge = Edge(step.token, rules.gap(scenario, step))
                if edge not in node.children:
                    node.children[edge] = Node()
                node = node.children[edge]
            if node.scenario is not None:
                raise ValueError(
                    f"сценарии {node.scenario!r} и {scenario.name!r} задают одну и ту же "
                    f"последовательность токенов {[step.token for step in pattern]} с теми же задержками"
                )
            node.scenario = scenario.name
    return starts


def graph_size(starts: dict[str, Node]) -> int:
    """Число вершин без корня."""

    def size(node: Node) -> int:
        return 1 + sum(size(child) for child in node.children.values())

    return sum(size(node) for node in starts.values())


def render(starts: dict[str, Node], max_gap: float) -> str:
    """Задержку показывает у рёбер, где она отличается от `max_gap` корня правил."""
    lines = ["(начало)"]

    def walk(items: list[tuple[str, Node]], indent: str) -> None:
        for index, (label, node) in enumerate(items):
            last = index == len(items) - 1
            mark = f"  ⇒ {node.scenario}" if node.scenario is not None else ""
            lines.append(f"{indent}{'└─ ' if last else '├─ '}{label}{mark}")
            children = [
                (edge.token if edge.max_gap == max_gap else f"{edge.token} [{edge.max_gap:g} с]", child)
                for edge, child in node.children.items()
            ]
            walk(children, indent + ("   " if last else "│  "))

    walk(list(starts.items()), "")
    return "\n".join(lines)
