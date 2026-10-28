"""Сценарии использования: правила и поток приходят текстом, наружу уходят view-модели.

Откуда взять текст — из файла или из формы — решает интерфейс.
"""

import json
from collections.abc import Iterable, Iterator
from typing import NamedTuple

from rule_engine.application.stream import parse_stream
from rule_engine.application.views import (
    EventView,
    Frame,
    GraphView,
    MatchView,
    MissView,
    NodeView,
    SetView,
    SlotView,
    TokenEdge,
    WaitingView,
)
from rule_engine.domain.engine import Engine, Miss, Step, Waiting
from rule_engine.domain.graph import Cursor, Edge, Node, Start
from rule_engine.domain.rules import RuleSet
from rule_engine.domain.unordered import Unordered


class Trace(NamedTuple):
    graph: GraphView
    frames: Iterator[Frame]  # кадр на событие; ошибка в строке вылетает вместо её кадра


class Index(NamedTuple):
    """id вершин и наборов: по ним снимок совпадений ссылается на `GraphView`."""

    nodes: dict[Node, int]
    sets: dict[Unordered, int]


class RuleEditor:
    """Набор правил с готовым движком: новые сценарии дописываются в его граф."""

    def __init__(self, rules_text: str) -> None:
        self._document = json.loads(rules_text)
        self._rules = RuleSet.from_dict(self._document)
        self._engine = Engine(self._rules)

    @property
    def rules_text(self) -> str:
        return json.dumps(self._document, ensure_ascii=False, indent=2)

    def graph(self) -> GraphView:
        return describe(self._engine, self._rules)[0]

    def add_scenario(self, scenario_text: str) -> None:
        scenario = json.loads(scenario_text)
        if not isinstance(scenario, dict):
            raise ValueError("сценарий должен быть объектом JSON")
        vocabulary = self._engine.vocabulary
        self._engine.add_scenario(scenario)
        self._document["scenarios"].append(scenario)
        self._document["vocabulary"].extend(sorted(self._engine.vocabulary - vocabulary))


class RuleEngineService:
    def editor(self, rules_text: str) -> RuleEditor:
        return RuleEditor(rules_text)

    def graph(self, rules_text: str) -> GraphView:
        rules = parse_rules(rules_text)
        return describe(Engine(rules), rules)[0]

    def trace(self, rules_text: str, lines: Iterable[str]) -> Trace:
        """Прогон потока. Кадры идут генератором: CLI печатает их по ходу, веб собирает все."""
        rules = parse_rules(rules_text)
        engine = Engine(rules)
        graph, index = describe(engine, rules)
        return Trace(graph, frames(engine, index, lines))


def parse_rules(text: str) -> RuleSet:
    return RuleSet.from_dict(json.loads(text))


def frames(engine: Engine, index: Index, lines: Iterable[str]) -> Iterator[Frame]:
    for event in parse_stream(lines):
        step = Step(engine.position, event)
        matches = engine.push(event)
        yield Frame(
            event_view(step),
            tuple(
                MatchView(match.scenario, tuple(map(event_view, match.steps)), match.confidence)
                for match in matches
            ),
            tuple(waiting_view(item, index) for item in engine.snapshot()),
            tuple(miss_view(miss, index) for miss in engine.misses),
        )


def describe(engine: Engine, rules: RuleSet) -> tuple[GraphView, Index]:
    """Граф движка как view; вершины нумеруются в порядке обхода, как их печатает CLI."""
    nodes: dict[Node, int] = {}

    def edge_view(edge: Start | Edge) -> TokenEdge:
        min_score = None if edge.min_score == rules.min_score else edge.min_score
        if isinstance(edge, Start) or edge.max_gap == rules.max_gap:
            return TokenEdge(edge.token, None, min_score)
        return TokenEdge(edge.token, edge.max_gap, min_score)

    def node_view(node: Node) -> NodeView:
        nodes[node] = len(nodes)
        children = tuple(node_view(child) for child in node.children.values())
        return NodeView(nodes[node], tuple(map(edge_view, node.segment)), node.scenario, children)

    roots = tuple(node_view(node) for node in engine.starts.values())
    sets = {pattern: number for number, pattern in enumerate(engine.unordered)}
    set_views = tuple(
        SetView(
            sets[pattern],
            pattern.scenario,
            tuple(
                SlotView(slot.token, None if slot.min_score == rules.min_score else slot.min_score)
                for slot in pattern.slots
            ),
            None if pattern.max_gap == rules.max_gap else pattern.max_gap,
        )
        for pattern in engine.unordered
    )
    graph = GraphView(roots, set_views, tuple(sorted(engine.vocabulary)))
    return graph, Index(nodes, sets)


def event_view(step: Step) -> EventView:
    return EventView(step.position, step.event.token, step.event.score, step.event.time)


def waiting_view(item: Waiting, index: Index) -> WaitingView:
    steps = tuple(map(event_view, item.partial.steps))
    confidence = item.partial.confidence
    if isinstance(item.place, Cursor):
        node = index.nodes[item.place.node]
        return WaitingView("node", node, frozenset(), steps, confidence, item.until, item.place.offset)
    pattern, filled = item.place
    return WaitingView("set", index.sets[pattern], filled, steps, confidence, item.until)


def miss_view(miss: Miss, index: Index) -> MissView:
    steps = () if miss.partial is None else tuple(map(event_view, miss.partial.steps))
    if miss.place is None:
        kind, place, filled = None, None, frozenset()
    elif isinstance(miss.place, Cursor):
        kind, place, filled = "node", index.nodes[miss.place.node], frozenset()
    else:
        pattern, filled = miss.place
        kind, place = "set", index.sets[pattern]
    offset = miss.place.offset if isinstance(miss.place, Cursor) else 0
    return MissView(miss.reason, kind, place, filled, steps, miss.scenario, miss.need, miss.got, offset)
