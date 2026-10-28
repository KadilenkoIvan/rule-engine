"""Префиксный граф сценариев: правила с общим началом делят вершины.

В граф входят сценарии с `order: "sequence"`; шаблоны `any` собирает unordered.py.
"""

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


@dataclass(eq=False)
class Node:
    """Сравнивается по identity: вершина служит ключом для активных совпадений."""

    # шаги с одним токеном, но разной задержкой или порогом — разные рёбра: иначе одному совпадению
    # в общей вершине пришлось бы ждать по срокам и порогам разных сценариев
    children: dict[Edge, "Node"] = field(default_factory=dict)
    # рёбра-тишины отдельно: их срок ведёт колесо таймеров, а не обход вершины на событии
    silences: dict[Silence, "Node"] = field(default_factory=dict)
    scenario: str | None = None  # сценарий, один из шаблонов которого завершается в этой вершине
    # (длина шаблона, порог общей уверенности его сценария) для шаблонов, которые кончаются ниже
    # вершины: по ним движок снимает совпадение, которое не дотянет ни до одного порога. Длина —
    # без шагов-тишин: в общую уверенность они не входят
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
    paths: list[tuple[Start, tuple[Edge | Silence, ...]]] = []
    for first, *rest in scenario.patterns:
        start = Start(first.token, rules.threshold(scenario, first))
        edges = tuple(edge_of(rules, scenario, step) for step in rest)
        paths.append((start, edges))
        node = starts.get(start)
        for edge in edges:
            if node is None:
                break
            node = branches(node, edge).get(edge)
        if node is not None and node.scenario is not None:
            tokens = [start.token, *map(edge_label, edges)]
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
            children = branches(node, edge)
            if edge not in children:
                children[edge] = Node()
            node = children[edge]
            nodes.append(node)
        node.scenario = scenario.name
        length = 1 + sum(isinstance(edge, Edge) for edge in edges)
        for above in nodes[:-1]:
            above.ahead.add((length, threshold))


def edge_of(rules: RuleSet, scenario: Scenario, step: PatternStep | SilenceStep) -> Edge | Silence:
    """Ребро шага с действующими задержкой и порогом."""
    if isinstance(step, SilenceStep):
        return Silence(
            tuple(sorted(step.absent)), step.duration, step.before, rules.threshold(scenario, step)
        )
    return Edge(step.token, rules.gap(scenario, step), rules.threshold(scenario, step))


def branches(node: Node, edge: Edge | Silence) -> dict:
    """Рёбра вершины того же вида, что и `edge`."""
    return node.silences if isinstance(edge, Silence) else node.children


def edge_label(edge: Edge | Silence) -> str:
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
    """Число вершин без корня."""

    def size(node: Node) -> int:
        children = (*node.children.values(), *node.silences.values())
        return 1 + sum(size(child) for child in children)

    return sum(size(node) for node in starts.values())

