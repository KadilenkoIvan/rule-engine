"""Префиксный граф сценариев: правила с общим началом делят вершины.

В граф входят сценарии с `order: "sequence"`; шаблоны `any` собирает unordered.py.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

from rule_engine.rules import PatternStep, RuleSet, SilenceStep, step_names
from rule_engine.unordered import Unordered


class Start(NamedTuple):
    """Ребро из корня. У первого шага задержки нет: новое совпадение её не проверяет."""

    token: str
    min_score: float  # порог score токена


class Edge(NamedTuple):
    token: str
    max_gap: float  # секунды от предыдущего шага
    min_score: float  # порог score токена


class Silence(NamedTuple):
    """Ребро-тишина: окно `window` секунд после предыдущего шага (и столько же до него) без
    токенов `absent` со score не ниже порога."""

    absent: tuple[str, ...]  # отсортированы: тот же набор в другом порядке — то же ребро
    window: float  # секунды
    min_score: float  # порог запрещённых токенов: токен ниже него тишину не нарушает


@dataclass(eq=False)
class Node:
    """Сравнивается по identity: вершина служит ключом для активных совпадений."""

    # шаги с одним токеном, но разной задержкой или порогом — разные рёбра: иначе одному совпадению
    # в общей вершине пришлось бы ждать по срокам и порогам разных сценариев
    children: dict[Edge, "Node"] = field(default_factory=dict)
    silences: dict[Silence, "Node"] = field(default_factory=dict)  # шаги-тишины из этой вершины
    scenario: str | None = None  # сценарий, один из шаблонов которого завершается в этой вершине
    # (длина шаблона, порог общей уверенности его сценария) для шаблонов, которые кончаются ниже
    # вершины: по ним движок снимает совпадение, которое не дотянет ни до одного порога. Длина —
    # число шагов с токенами: тишина в уверенность не входит
    ahead: set[tuple[int, float]] = field(default_factory=set)


def build_graph(rules: RuleSet) -> dict[Start, Node]:
    """Рёбра из корня: токен и порог первого шага → вершина.

    Дальше ребро — токен, действующие задержка и порог шага: общее начало у сценариев общее,
    только если настройки на нём совпадают. Шаг-тишина — отдельное ребро `Silence`.
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
                if isinstance(step, SilenceStep):
                    silence = Silence(
                        tuple(sorted(step.absent)), step.window, rules.threshold(scenario, step)
                    )
                    if silence not in node.silences:
                        node.silences[silence] = Node()
                    node = node.silences[silence]
                else:
                    edge = Edge(
                        step.token, rules.gap(scenario, step), rules.threshold(scenario, step)
                    )
                    if edge not in node.children:
                        node.children[edge] = Node()
                    node = node.children[edge]
                nodes.append(node)
            if node.scenario is not None:
                raise ValueError(
                    f"сценарии {node.scenario!r} и {scenario.name!r} задают одну и ту же "
                    f"последовательность шагов {step_names(pattern)} с теми же задержками и "
                    "порогами"
                )
            node.scenario = scenario.name
            length = sum(isinstance(step, PatternStep) for step in pattern)
            for above in nodes[:-1]:
                above.ahead.add((length, rules.confidence_threshold(scenario)))
    return starts


def graph_size(starts: dict[Start, Node]) -> int:
    """Число вершин без корня."""

    def size(node: Node) -> int:
        return 1 + sum(size(child) for child in (*node.children.values(), *node.silences.values()))

    return sum(size(node) for node in starts.values())


def render(starts: dict[Start, Node], unordered: list[Unordered], rules: RuleSet) -> str:
    """Бор, под ним шаблоны `any`. Задержку и порог показывает, где они отличаются от `max_gap` и
    `min_score` корня."""
    lines = ["(начало)"]

    def label(edge: Start | Edge | Silence) -> str:
        notes = []
        if isinstance(edge, Edge) and edge.max_gap != rules.max_gap:
            notes.append(f"{edge.max_gap:g} с")
        if edge.min_score != rules.min_score:
            notes.append(f"≥ {edge.min_score:g}")
        if isinstance(edge, Silence):
            name = f"тишина {edge.window:g} с без {', '.join(edge.absent)}"
        else:
            name = edge.token
        return f"{name} [{', '.join(notes)}]" if notes else name

    def walk(edges: list[tuple[Start | Edge | Silence, Node]], indent: str) -> None:
        for index, (edge, node) in enumerate(edges):
            last = index == len(edges) - 1
            mark = f"  ⇒ {node.scenario}" if node.scenario is not None else ""
            lines.append(f"{indent}{'└─ ' if last else '├─ '}{label(edge)}{mark}")
            walk(
                [*node.children.items(), *node.silences.items()],
                indent + ("   " if last else "│  "),
            )

    def label_set(pattern: Unordered) -> str:
        # одинаковые у всех шагов пороги показаны один раз, после набора
        common = len({slot.min_score for slot in pattern.slots}) == 1
        tokens = ", ".join(
            slot.token
            if common or slot.min_score == rules.min_score
            else f"{slot.token} [≥ {slot.min_score:g}]"
            for slot in pattern.slots
        )
        notes = []
        if len(pattern.slots) > 1 and pattern.max_gap != rules.max_gap:
            notes.append(f"{pattern.max_gap:g} с")
        if common and pattern.slots[0].min_score != rules.min_score:
            notes.append(f"≥ {pattern.slots[0].min_score:g}")
        return f"{{{tokens}}} [{', '.join(notes)}]" if notes else f"{{{tokens}}}"

    walk(list(starts.items()), "")
    if unordered:
        lines.append("(в любом порядке)")
        for index, pattern in enumerate(unordered):
            branch = "└─ " if index == len(unordered) - 1 else "├─ "
            lines.append(f"{branch}{label_set(pattern)}  ⇒ {pattern.scenario}")
    return "\n".join(lines)
