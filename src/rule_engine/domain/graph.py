"""Radix trie: цепочки шагов без развилок и завершений хранятся одним ребром."""

from dataclasses import dataclass, field
from typing import Iterator, NamedTuple

from rule_engine.domain.rules import RuleSet, Scenario


class Start(NamedTuple):
    token: str
    min_score: float


class Edge(NamedTuple):
    token: str
    max_gap: float
    min_score: float


@dataclass(eq=False, slots=True)
class Node:
    """Конец сжатого ребра; segment — все ведущие в него шаги."""

    segment: tuple[Start | Edge, ...]
    children: dict[Edge, "Node"] = field(default_factory=dict)
    scenario: str | None = None
    ahead: set[tuple[int, float]] = field(default_factory=set)
    # Все достижимые завершения, включая эту вершину: для позиций внутри segment.
    through: set[tuple[int, float]] = field(default_factory=set)


class Cursor(NamedTuple):
    """Позиция после offset принятых шагов сжатого ребра (от 1 до его длины)."""

    node: Node
    offset: int

    @property
    def scenario(self) -> str | None:
        return self.node.scenario if self.offset == len(self.node.segment) else None

    @property
    def ahead(self) -> set[tuple[int, float]]:
        return self.node.through if self.offset < len(self.node.segment) else self.node.ahead

    def transitions(self) -> Iterator[tuple[Edge, "Cursor"]]:
        if self.offset < len(self.node.segment):
            edge = self.node.segment[self.offset]
            yield edge, Cursor(self.node, self.offset + 1)
        else:
            for edge, child in self.node.children.items():
                yield edge, Cursor(child, 1)


def build_graph(rules: RuleSet) -> dict[Start, Node]:
    starts: dict[Start, Node] = {}
    for scenario in rules.scenarios:
        insert_scenario(starts, rules, scenario)
    return starts


def insert_scenario(
    starts: dict[Start, Node], rules: RuleSet, scenario: Scenario
) -> list[tuple[Node, Node, int]]:
    """Вставляет пути, расщепляя рёбра. Возвращает расщепления для переноса живых курсоров.

    Старый объект Node остаётся концом суффикса; новые вершины появляются только в местах
    развилок и завершений. Все конфликты проверяются до первой модификации графа.
    """
    if scenario.order != "sequence":
        return []
    paths = [
        (Start(first.token, rules.threshold(scenario, first)),)
        + tuple(
            Edge(step.token, rules.gap(scenario, step), rules.threshold(scenario, step))
            for step in rest
        )
        for first, *rest in scenario.patterns
    ]
    for path in paths:
        node = starts.get(path[0])
        remaining = path
        while node is not None:
            if remaining[: len(node.segment)] != node.segment:
                break
            remaining = remaining[len(node.segment) :]
            if not remaining:
                if node.scenario is not None:
                    raise ValueError(
                        f"сценарии {node.scenario!r} и {scenario.name!r} задают одну и ту же "
                        f"последовательность токенов {[step.token for step in path]} "
                        "с теми же задержками и порогами"
                    )
                break
            node = node.children.get(remaining[0])

    splits: list[tuple[Node, Node, int]] = []
    threshold = rules.confidence_threshold(scenario)
    for path in paths:
        edges = starts
        remaining = path
        ancestors = []
        target = (len(path), threshold)
        while True:
            key = remaining[0]
            node = edges.get(key)
            if node is None:
                node = Node(remaining)
                edges[key] = node
                break
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
                prefix.children[node.segment[0]] = node
                edges[key] = prefix
                splits.append((node, prefix, common))
                node = prefix
            remaining = remaining[len(node.segment) :]
            if not remaining:
                break
            ancestors.append(node)
            edges = node.children
        node.scenario = scenario.name
        node.through.add(target)
        for above in ancestors:
            above.ahead.add(target)
            above.through.add(target)
    return splits


def graph_size(starts: dict[Start, Node]) -> int:
    """Число физических вершин, без виртуального корня."""

    def size(node: Node) -> int:
        return 1 + sum(size(child) for child in node.children.values())

    return sum(size(node) for node in starts.values())
