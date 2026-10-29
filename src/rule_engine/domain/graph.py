"""Radix trie: token chains are compressed between forks, pattern ends and silences.

The graph holds scenarios with `order: "sequence"`; `any` patterns are built by unordered.py.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

from rule_engine.domain.rules import PatternStep, RuleSet, Scenario, SilenceStep


class Start(NamedTuple):
    """Edge from the root. The first step has no gap: a new match does not check it."""

    token: str
    min_score: float  # порог score токена


class Edge(NamedTuple):
    """Token edge with its effective gap and threshold."""

    token: str
    max_gap: float  # секунды от предыдущего шага
    min_score: float  # порог score токена


class Silence(NamedTuple):
    """Silence edge: a `duration` window without `tokens` at or above the threshold."""

    tokens: tuple[str, ...]  # по алфавиту: ребро не зависит от порядка в JSON
    duration: float  # секунды вперёд от предыдущего шага
    before: float  # секунды назад от предыдущего шага; 0 — назад не смотрим
    min_score: float  # порог score запрещённых токенов


class Targets:
    """Pattern ends below a position: the lowest confidence threshold per pattern length.

    The engine prunes a match that reaches none of them. For patterns of one length L,
    `product ** (1 / L) >= threshold` holds for some pattern only if it holds for the lowest
    threshold, so one pair per length is enough.
    """

    __slots__ = ("lowest",)

    def __init__(self) -> None:
        self.lowest: dict[int, float] = {}  # длина шаблона → наименьший порог

    def add(self, target: tuple[int, float]) -> None:
        """Adds a pattern end: (length, confidence threshold)."""
        length, threshold = target
        if threshold < self.lowest.get(length, math.inf):
            self.lowest[length] = threshold

    def update(self, other: "Targets") -> None:
        """Adds all pattern ends of `other`."""
        for target in other:
            self.add(target)

    def __iter__(self) -> Iterator[tuple[int, float]]:
        return iter(self.lowest.items())

    def __len__(self) -> int:
        return len(self.lowest)

    def reaches(self, product: float) -> bool:
        """Whether a match with this score product can reach at least one threshold."""
        return any(product ** (1 / length) >= threshold for length, threshold in self.lowest.items())


@dataclass(eq=False, slots=True)
class Node:
    """End of a compressed edge; compared by identity, cursors refer to this object."""

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
    """Position after `offset` steps of the node segment; a silence is always a segment alone."""

    node: Node
    offset: int

    @property
    def scenario(self) -> str | None:
        """Scenario whose pattern ends here; None inside a segment."""
        return self.node.scenario if self.offset == len(self.node.segment) else None

    @property
    def ahead(self) -> Targets:
        """Pattern ends reachable from here."""
        return self.node.through if self.offset < len(self.node.segment) else self.node.ahead

    @property
    def max_gap(self) -> float:
        """Largest gap of the transitions: after it a match cannot continue from here."""
        if self.offset < len(self.node.segment):
            return self.node.segment[self.offset].max_gap
        return self.node.max_gap

    def transitions(self) -> Iterator[tuple[Edge, "Cursor"]]:
        """Token transitions from here and the positions behind them."""
        if self.offset < len(self.node.segment):
            yield self.node.segment[self.offset], Cursor(self.node, self.offset + 1)
        else:
            for edge, child in self.node.children.items():
                yield edge, Cursor(child, 1)

    def follow(self, edge: Edge) -> "Cursor":
        """Position behind transition `edge` from here."""
        if self.offset < len(self.node.segment):
            return Cursor(self.node, self.offset + 1)
        return Cursor(self.node.children[edge], 1)

    def silences(self) -> Iterator[Silence]:
        """Silence edges from here: only at the end of the segment."""
        if self.offset == len(self.node.segment):
            yield from self.node.silences


def build_graph(rules: RuleSet) -> dict[Start, Node]:
    """Builds the trie of `sequence` scenarios.

    Edges carry the token, effective gap and threshold of a step, so scenarios share a common
    start only if its settings are equal.

    Returns:
        Root edges (token and threshold of the first step) → nodes.
    """
    starts: dict[Start, Node] = {}
    for scenario in rules.scenarios:
        insert_scenario(starts, rules, scenario)
    return starts


class Changes(NamedTuple):
    """What an insert changed in the trie: the engine moves and extends its indexes by it."""

    splits: list[tuple[Node, Node, int]]  # (вершина, теперь суффикс; её префикс; длина префикса)
    starts: list[Start]  # новые рёбра из корня
    links: list[tuple[Node, Edge]]  # новые токеновые рёбра из конца сегмента вершины


def attach(node: Node, edge: Edge | Silence, child: Node) -> None:
    """Edge from the end of `node`'s segment; the segment end keeps the largest edge gap."""
    if isinstance(edge, Silence):
        node.silences[edge] = child
    else:
        node.children[edge] = child
        node.max_gap = max(node.max_gap, edge.max_gap)


def insert_scenario(starts: dict[Start, Node], rules: RuleSet, scenario: Scenario) -> Changes:
    """Adds the scenario's paths to the trie.

    Conflicts of whole patterns are checked before the trie changes, by the old rules. A split
    node stays the end of its suffix, so its silence edges and open windows stay valid.

    Returns:
        What changed: the engine moves active cursors and extends its indexes by it.

    Raises:
        ValueError: another scenario has the same path with the same gaps and thresholds.
    """
    changes = Changes([], [], [])
    if scenario.order != "sequence":
        return changes
    paths: list[tuple[Start | Edge | Silence, ...]] = []
    for first, *rest in scenario.patterns:
        start = Start(first.token, rules.threshold(scenario, first))
        path = (start, *(edge_of(rules, scenario, step) for step in rest))
        paths.append(path)
        _check_conflict(starts, path, scenario)
    threshold = rules.confidence_threshold(scenario)
    for path in paths:
        _insert_path(starts, path, scenario.name, threshold, changes)
    return changes


def _check_conflict(
    starts: dict[Start, Node], path: tuple[Start | Edge | Silence, ...], scenario: Scenario
) -> None:
    """Raises if `path` already ends a pattern of another scenario."""
    node = starts.get(path[0])
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


def _insert_path(
    starts: dict[Start, Node],
    path: tuple[Start | Edge | Silence, ...],
    name: str,
    threshold: float,
    changes: Changes,
) -> None:
    """Inserts one path and marks its end with scenario `name`.

    Args:
        starts: root edges of the trie.
        path: root edge and edges of the pattern steps.
        name: the scenario.
        threshold: its confidence threshold.
        changes: changes of the trie; filled here.
    """
    remaining = path
    parent: Node | None = None
    ancestors = []
    target = (sum(not isinstance(edge, Silence) for edge in path), threshold)
    while True:
        key = remaining[0]
        children = starts if parent is None else branches(parent, key)
        node = children.get(key)
        if node is None:
            node = _grow(starts, parent, remaining, changes)
        else:
            node = _split(node, remaining, children, key, changes)
        remaining = remaining[len(node.segment) :]
        if not remaining:
            break
        ancestors.append(node)
        parent = node
    node.scenario = name
    node.through.add(target)
    for above in ancestors:
        above.ahead.add(target)
        above.through.add(target)


def _grow(
    starts: dict[Start, Node],
    parent: Node | None,
    remaining: tuple[Start | Edge | Silence, ...],
    changes: Changes,
) -> Node:
    """New node for the start of `remaining` under `parent` (None — the root)."""
    key = remaining[0]
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
    return node


def _split(
    node: Node,
    remaining: tuple[Start | Edge | Silence, ...],
    children: dict,
    key: Start | Edge | Silence,
    changes: Changes,
) -> Node:
    """Node for the common start of `node`'s segment and `remaining`.

    If they diverge inside the segment, the common part becomes a new prefix node under edge
    `key` in `children`, and `node` keeps the rest.
    """
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
    return node


def edge_of(rules: RuleSet, scenario: Scenario, step: PatternStep | SilenceStep) -> Edge | Silence:
    """Edge of a step with its effective gap and threshold."""
    if isinstance(step, SilenceStep):
        return Silence(
            tuple(sorted(step.absent)), step.duration, step.before, rules.threshold(scenario, step)
        )
    return Edge(step.token, rules.gap(scenario, step), rules.threshold(scenario, step))


def branches(node: Node, edge: Start | Edge | Silence) -> dict:
    """Edges of the node of the same kind as `edge`."""
    return node.silences if isinstance(edge, Silence) else node.children


def edge_label(edge: Start | Edge | Silence) -> str:
    """Edge as shown in error messages."""
    if isinstance(edge, Silence):
        before = f", назад {edge.before:g} с" if edge.before else ""
        return f"тишина {list(edge.tokens)} {edge.duration:g} с{before}"
    return edge.token


def longest_before(starts: dict[Start, Node]) -> float:
    """Longest backward silence window in the trie; 0 if no silence looks back."""
    return max((_longest_below(node) for node in starts.values()), default=0.0)


def _longest_below(node: Node) -> float:
    """Longest backward silence window below `node`."""
    befores = (
        max(edge.before, _longest_below(child))
        if isinstance(edge, Silence)
        else _longest_below(child)
        for edge, child in (*node.children.items(), *node.silences.items())
    )
    return max(befores, default=0.0)


def graph_size(starts: dict[Start, Node]) -> int:
    """Number of physical radix trie nodes, without the root."""
    return sum(_size(node) for node in starts.values())


def _size(node: Node) -> int:
    """Number of nodes in the subtree of `node`."""
    children = (*node.children.values(), *node.silences.values())
    return 1 + sum(_size(child) for child in children)
