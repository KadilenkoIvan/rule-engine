"""Сценарии `sequence` в проходе: совпадения стоят в вершинах бора и идут по его рёбрам."""

from collections.abc import Iterator
from itertools import islice

from rule_engine.domain.front import Front
from rule_engine.domain.graph import Edge, Node, Start, insert_scenario
from rule_engine.domain.matching import Event, Miss, Partial, Tick
from rule_engine.domain.rules import RuleSet, Scenario


class NodeFront(Front):
    """Совпадения в вершине бора. Срок у них — наибольшая задержка рёбер вершины."""

    __slots__ = ("_trie", "node")

    def __init__(self, trie: "TrieMatcher", node: Node) -> None:
        super().__init__()
        self._trie = trie
        self.node = node

    @property
    def place(self) -> Node:
        return self.node

    @property
    def gap(self) -> float:
        return self.node.max_gap

    def expired(self, partial: Partial, now: float) -> bool:
        # срок считается от последнего шага совпадения, а не от последнего токена потока
        return now - partial.last_time > self.node.max_gap

    def expiry_misses(self, partial: Partial, event: Event) -> list[Miss]:
        last = partial.last_time
        # есть ребро с токеном события: снятие объясняет опоздание на нём
        late = [
            Miss("late", child, partial, None, last + edge.max_gap, event.time)
            for edge, child in self.node.children.items()
            if edge.token == event.token
        ]
        deadline = last + self.node.max_gap
        return late or [Miss("expired", self.node, partial, None, deadline, event.time)]

    def release(self) -> None:
        del self._trie.fronts[self.node]


class TrieMatcher:
    """Сценарии `sequence`: бор, индексы его рёбер по токену и совпадения в вершинах.

    Событие проверяет только рёбра со своим токеном, а не все вершины, где ждут совпадения.
    """

    def __init__(self, rules: RuleSet) -> None:
        self.starts: dict[Start, Node] = {}
        self._start_index: dict[str, list[tuple[Start, Node]]] = {}  # токен → рёбра из корня
        self._edge_index: dict[str, list[tuple[Node, Edge, Node]]] = {}  # токен → рёбра бора
        self.fronts: dict[Node, NodeFront] = {}  # вершина → совпадения в ней, без пустых
        for scenario in rules.scenarios:
            self.add(rules, scenario)

    def add(self, rules: RuleSet, scenario: Scenario) -> None:
        """Дописывает пути сценария в бор, а его новые рёбра — в индексы.

        Новое ребро может удлинить задержку вершины. Срок её фронта в очереди тогда раньше
        настоящего, и фронт просто вернётся в очередь.
        """
        for parent, edge, child in insert_scenario(self.starts, rules, scenario):
            if parent is None:
                self._start_index.setdefault(edge.token, []).append((edge, child))
            else:
                self._edge_index.setdefault(edge.token, []).append((parent, edge, child))

    def front(self, node: Node) -> NodeFront:
        front = self.fronts.get(node)
        if front is None:
            front = self.fronts[node] = NodeFront(self, node)
        return front

    def advance(self, tick: Tick) -> None:
        """Продолжает ждущие совпадения по рёбрам с токеном события."""
        for parent, edge, child in self._edge_index.get(tick.event.token, ()):
            front = self.fronts.get(parent)
            if front is not None:
                self._cross(tick, front, edge, child)

    def start(self, tick: Tick) -> None:
        """Начинает новые совпадения по рёбрам из корня; задержку они не проверяют."""
        event = tick.event
        for start, child in self._start_index.get(event.token, ()):
            if event.score < start.min_score:
                tick.misses.append(Miss("score", child, None, None, start.min_score, event.score))
                continue
            partial = Partial((tick.step,), event.score)
            if child.scenario is not None:
                tick.complete(child.scenario, [partial])
            if child.promising(partial.product):
                tick.enter(self, child, partial)
            elif child.ahead:
                prune(tick, child, partial)

    def _cross(self, tick: Tick, front: NodeFront, edge: Edge, child: Node) -> None:
        """Продолжает совпадения фронта по ребру `edge` в вершину `child`."""
        event, step = tick.event, tick.step
        # фронт идёт от старых к свежим, поэтому в задержку ребра укладывается его хвост
        fresh = front.split(lambda partial: event.time - partial.last_time <= edge.max_gap)
        for partial in islice(front, fresh):
            deadline = partial.last_time + edge.max_gap
            tick.misses.append(Miss("late", child, partial, None, deadline, event.time))
        if fresh == len(front):
            return
        if event.score < edge.min_score:
            for partial in islice(front, fresh, None):
                tick.misses.append(Miss("score", child, partial, None, edge.min_score, event.score))
            return
        if child.scenario is not None:
            tick.complete(
                child.scenario, (partial.extend(step) for partial in islice(front, fresh, None))
            )
        # все продолжения входят в одну вершину с одним последним шагом, поэтому остаётся
        # одно — с наибольшим произведением, то есть первое в хвосте
        best = front[fresh].extend(step)
        if child.promising(best.product):
            tick.enter(self, child, best)
        if child.ahead:
            for partial in front.weakest(
                fresh, lambda partial: child.promising(partial.product * event.score)
            ):
                prune(tick, child, partial.extend(step))

    def all_fronts(self) -> Iterator[NodeFront]:
        return iter(self.fronts.values())


def prune(tick: Tick, node: Node, partial: Partial) -> None:
    """Отказ «отсеяно»: совпадение не дотянет ни до одного порога ниже вершины `node`."""
    length, threshold = node.nearest(partial.product)
    tick.misses.append(Miss("pruned", node, partial, None, threshold, partial.bound(length)))
