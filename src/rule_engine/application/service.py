"""Сценарии использования: правила и поток приходят текстом, наружу уходят view-модели.

Откуда взять текст — из файла или из формы — решает интерфейс.
"""

import json
import math
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
    SilenceEdge,
    SlotView,
    TokenEdge,
    WaitingView,
)
from rule_engine.domain.engine import Engine, Match, Miss, Step, Waiting, Window
from rule_engine.domain.graph import Edge, Node, Silence, Start
from rule_engine.domain.rules import RuleSet
from rule_engine.domain.unordered import Unordered


class Trace(NamedTuple):
    graph: GraphView
    # кадр на событие и на конец окна тишины; ошибка в строке вылетает вместо её кадра
    frames: Iterator[Frame]


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
    for item in parse_stream(lines):
        if isinstance(item, float):
            yield from timer_frames(engine, index, item)
            engine.advance(item)  # окна раньше метки уже закрыты: метка только двигает время
            continue
        yield from timer_frames(engine, index, item.time)
        step = Step(engine.position, item)
        yield frame(engine, index, item.time, event_view(step), engine.push(item))


def timer_frames(engine: Engine, index: Index, until: float) -> Iterator[Frame]:
    """Кадр на каждый конец окна тишины раньше `until`: время прошло без события.

    Движок закрывает окно, когда время станет больше его срока, поэтому время сдвигается на
    ближайшее число после срока, а кадр показывает сам срок.
    """
    while (deadline := engine.next_deadline()) is not None and deadline < until:
        matches = engine.advance(math.nextafter(deadline, math.inf))
        yield frame(engine, index, deadline, None, matches)


def frame(
    engine: Engine, index: Index, time: float, event: EventView | None, matches: list[Match]
) -> Frame:
    return Frame(
        time,
        event,
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

    def edge_view(edge: Start | Edge | Silence) -> TokenEdge | SilenceEdge:
        min_score = None if edge.min_score == rules.min_score else edge.min_score
        if isinstance(edge, Silence):
            return SilenceEdge(edge.tokens, edge.duration, edge.before, min_score)
        if isinstance(edge, Start) or edge.max_gap == rules.max_gap:
            return TokenEdge(edge.token, None, min_score)
        return TokenEdge(edge.token, edge.max_gap, min_score)

    def node_view(edge: Start | Edge | Silence, node: Node) -> NodeView:
        nodes[node] = len(nodes)
        branches = (*node.children.items(), *node.silences.items())
        children = tuple(node_view(child_edge, child) for child_edge, child in branches)
        return NodeView(nodes[node], edge_view(edge), node.scenario, children)

    roots = tuple(node_view(start, node) for start, node in engine.starts.items())
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
    if isinstance(item.place, Node):
        node = index.nodes[item.place]
        return WaitingView("node", node, frozenset(), steps, confidence, item.until)
    if isinstance(item.place, Window):
        node = index.nodes[item.place.child]
        return WaitingView("silence", node, frozenset(), steps, confidence, item.until)
    pattern, filled = item.place
    return WaitingView("set", index.sets[pattern], filled, steps, confidence, item.until)


def miss_view(miss: Miss, index: Index) -> MissView:
    steps = () if miss.partial is None else tuple(map(event_view, miss.partial.steps))
    if miss.place is None:
        kind, place, filled = None, None, frozenset()
    elif isinstance(miss.place, Node):
        kind, place, filled = "node", index.nodes[miss.place], frozenset()
    elif isinstance(miss.place, Window):
        kind, place, filled = "silence", index.nodes[miss.place.child], frozenset()
    else:
        pattern, filled = miss.place
        kind, place = "set", index.sets[pattern]
    return MissView(miss.reason, kind, place, filled, steps, miss.scenario, miss.need, miss.got)
