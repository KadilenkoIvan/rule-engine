"""Use cases: rules and streams come as text, view models go out.

Where the text comes from, a file or a form, is up to the interface.
"""

import json
import math
from collections.abc import Callable, Iterable, Iterator
from threading import Lock
from time import monotonic
from typing import NamedTuple

from rule_engine.application.ports import LiveListener
from rule_engine.application.stream import check_token, parse_stream
from rule_engine.application.views import (
    EventView,
    Frame,
    GraphView,
    HeldView,
    MatchView,
    MissView,
    NodeView,
    SetView,
    SilenceEdge,
    SlotView,
    TokenEdge,
    WaitingView,
)
from rule_engine.domain.engine import Engine, Held, Match, Miss, Step, Waiting, Window
from rule_engine.domain.graph import Cursor, Edge, Node, Silence, Start
from rule_engine.domain.live import LiveEngine, Update
from rule_engine.domain.rules import RuleSet
from rule_engine.domain.unordered import Unordered


class Trace(NamedTuple):
    """Graph of the rules and the frames of a stream run."""

    graph: GraphView
    # кадр на событие и на конец окна тишины; ошибка в строке вылетает вместо её кадра
    frames: Iterator[Frame]


class Index(NamedTuple):
    """Ids of nodes and sets: snapshots refer to `GraphView` by them."""

    nodes: dict[Cursor, int]
    sets: dict[Unordered, int]


class RuleEditor:
    """Rule set with a ready engine: new scenarios are added to its graph."""

    def __init__(self, rules_text: str) -> None:
        self._document = json.loads(rules_text)
        self._rules = RuleSet.from_dict(self._document)
        self._engine = Engine(self._rules)

    @property
    def rules_text(self) -> str:
        """Rules as JSON text, added scenarios and their tokens included."""
        return json.dumps(self._document, ensure_ascii=False, indent=2)

    def graph(self) -> GraphView:
        """Graph of the current rules."""
        return describe(self._engine, self._rules)[0]

    def add_scenario(self, scenario_text: str) -> None:
        """Adds a scenario given as JSON text.

        Raises:
            ValueError: the text is not a JSON object, or the scenario is invalid.
        """
        scenario = json.loads(scenario_text)
        if not isinstance(scenario, dict):
            raise ValueError("сценарий должен быть объектом JSON")
        vocabulary = self._engine.vocabulary
        self._engine.add_scenario(scenario)
        self._document["scenarios"].append(scenario)
        self._document["vocabulary"].extend(sorted(self._engine.vocabulary - vocabulary))


class LiveSession:
    """Live mode: tokens come «now», the engine's timer thread moves time.

    Time is seconds from the session start. Updates of `LiveEngine` may overtake each other,
    so the session holds an early one and passes frames to the listener strictly by number.
    """

    def __init__(self, rules: RuleSet, listener: LiveListener, clock: Callable[[], float]) -> None:
        self._listener = listener
        self._lock = Lock()  # кадры уходят слушателю по одному
        self._next = 1  # номер обновления, которое уйдёт слушателю следующим
        self._early: dict[int, Update] = {}  # обновления, обогнавшие предыдущие
        start = clock()
        self._engine = LiveEngine(rules, self._receive, lambda: clock() - start)
        # обновлений ещё нет: окон нет, и токен не отправить, пока сессия не создана
        self._graph, self._index = describe(self._engine, rules)

    @property
    def graph(self) -> GraphView:
        """Graph of the session rules."""
        return self._graph

    def push(self, token: str, score: float) -> None:
        """Pushes a token «now»; its frame goes to the listener.

        Raises:
            ValueError: the token cannot be a stream line, or the session is closed.
        """
        check_token(token)
        self._engine.push_now(token, score)

    def close(self) -> None:
        """Stops the timer thread: no more frames."""
        self._engine.close()

    def _receive(self, update: Update) -> None:
        """Update of the engine: passes the frames that are next in order to the listener."""
        with self._lock:
            self._early[update.number] = update
            while self._next in self._early:
                ready = self._early.pop(self._next)
                self._next += 1
                event = None if ready.step is None else event_view(ready.step)
                frame = view_frame(
                    self._index,
                    ready.time,
                    event,
                    ready.matches,
                    ready.waiting,
                    ready.held,
                    ready.misses,
                )
                self._listener.on_frame(frame)


class RuleEngineService:
    """Use cases of the interfaces; rules come as JSON text."""

    def editor(self, rules_text: str) -> RuleEditor:
        """Rule editor with a ready engine."""
        return RuleEditor(rules_text)

    def graph(self, rules_text: str) -> GraphView:
        """Graph of the rules."""
        rules = parse_rules(rules_text)
        return describe(Engine(rules), rules)[0]

    def trace(self, rules_text: str, lines: Iterable[str]) -> Trace:
        """Stream run; frames are a generator: the CLI prints them on the go, the web collects."""
        rules = parse_rules(rules_text)
        engine = Engine(rules)
        graph, index = describe(engine, rules)
        return Trace(graph, frames(engine, index, lines))

    def live(
        self,
        rules_text: str,
        listener: LiveListener,
        clock: Callable[[], float] = monotonic,
    ) -> LiveSession:
        """Live session: a frame after every token and every timer move."""
        return LiveSession(parse_rules(rules_text), listener, clock)


def parse_rules(text: str) -> RuleSet:
    """Rules from JSON text."""
    return RuleSet.from_dict(json.loads(text))


def frames(engine: Engine, index: Index, lines: Iterable[str]) -> Iterator[Frame]:
    """Frames of a stream run: one per event and per deadline passed without an event."""
    for item in parse_stream(lines):
        if isinstance(item, float):
            yield from timer_frames(engine, index, item)
            engine.advance(item)  # окна раньше метки уже закрыты: метка только двигает время
            continue
        yield from timer_frames(engine, index, item.time)
        step = Step(engine.position, item)
        yield frame(engine, index, item.time, event_view(step), engine.push(item))


def timer_frames(engine: Engine, index: Index, until: float) -> Iterator[Frame]:
    """Frames for deadlines earlier than `until`: time passed without an event.

    The engine acts once time is greater than a deadline, so time moves to the next float after
    it, while the frame shows the deadline itself.
    """
    while (deadline := engine.next_deadline()) is not None and deadline < until:
        matches = engine.advance(math.nextafter(deadline, math.inf))
        yield frame(engine, index, deadline, None, matches)


def frame(
    engine: Engine, index: Index, time: float, event: EventView | None, matches: list[Match]
) -> Frame:
    """Frame of the engine state after an event or a move."""
    return view_frame(
        index, time, event, matches, engine.snapshot(), engine.held, engine.misses
    )


def view_frame(
    index: Index,
    time: float,
    event: EventView | None,
    matches: Iterable[Match],
    waiting: Iterable[Waiting],
    held: Iterable[Held],
    misses: Iterable[Miss],
) -> Frame:
    """Frame of what the engine returned: shared by stream runs and the live mode."""
    return Frame(
        time,
        event,
        tuple(map(match_view, matches)),
        tuple(waiting_view(item, index) for item in waiting),
        tuple(map(held_view, held)),
        tuple(miss_view(miss, index) for miss in misses),
    )


def describe(engine: Engine | LiveEngine, rules: RuleSet) -> tuple[GraphView, Index]:
    """Graph of the engine as a view; nodes are numbered in walk order, as the CLI prints them.

    Returns:
        The graph view and the ids of its nodes and sets.
    """
    nodes: dict[Cursor, int] = {}
    roots = tuple(_node_view(node, 1, nodes, rules) for node in engine.starts.values())
    sets = {pattern: number for number, pattern in enumerate(engine.unordered)}
    set_views = tuple(_set_view(sets[pattern], pattern, rules) for pattern in engine.unordered)
    graph = GraphView(roots, set_views, tuple(sorted(engine.vocabulary)))
    return graph, Index(nodes, sets)


def _node_view(node: Node, offset: int, nodes: dict[Cursor, int], rules: RuleSet) -> NodeView:
    """View of the position after `offset` steps of `node`, with everything below it.

    Args:
        node: the trie node.
        offset: steps of its segment passed.
        nodes: position → id; filled here in walk order.
        rules: the rules: settings equal to the root's are not shown.
    """
    # Интерфейсы получают прежнее логическое дерево, не физические radix-рёбра.
    cursor = Cursor(node, offset)
    nodes[cursor] = len(nodes)
    if offset < len(node.segment):
        children = (_node_view(node, offset + 1, nodes, rules),)
    else:
        branches = (*node.children.values(), *node.silences.values())
        children = tuple(_node_view(child, 1, nodes, rules) for child in branches)
    edge = _edge_view(node.segment[offset - 1], rules)
    return NodeView(nodes[cursor], edge, cursor.scenario, children)


def _edge_view(edge: Start | Edge | Silence, rules: RuleSet) -> TokenEdge | SilenceEdge:
    """View of an edge; gap and threshold equal to the root's become None."""
    min_score = None if edge.min_score == rules.min_score else edge.min_score
    if isinstance(edge, Silence):
        return SilenceEdge(edge.tokens, edge.duration, edge.before, min_score)
    if isinstance(edge, Start) or edge.max_gap == rules.max_gap:
        return TokenEdge(edge.token, None, min_score)
    return TokenEdge(edge.token, edge.max_gap, min_score)


def _set_view(number: int, pattern: Unordered, rules: RuleSet) -> SetView:
    """View of an `any` pattern; gap and thresholds equal to the root's become None."""
    return SetView(
        number,
        pattern.scenario,
        tuple(
            SlotView(slot.token, None if slot.min_score == rules.min_score else slot.min_score)
            for slot in pattern.slots
        ),
        None if pattern.max_gap == rules.max_gap else pattern.max_gap,
    )


def event_view(step: Step) -> EventView:
    """View of a stream event."""
    return EventView(step.position, step.event.token, step.event.score, step.event.time)


def match_view(match: Match) -> MatchView:
    """View of a fired scenario."""
    return MatchView(
        match.scenario, tuple(map(event_view, match.steps)), match.confidence, match.released
    )


def held_view(item: Held) -> HeldView:
    """View of a completion waiting for release."""
    return HeldView(
        item.scenario,
        tuple(map(event_view, item.partial.steps)),
        item.partial.confidence,
        item.until,
        item.frozen,
    )


def waiting_view(item: Waiting, index: Index) -> WaitingView:
    """View of a partial match; its place becomes an id of `index`."""
    steps = tuple(map(event_view, item.partial.steps))
    confidence = item.partial.confidence
    if isinstance(item.place, Cursor):
        node = index.nodes[item.place]
        return WaitingView("node", node, frozenset(), steps, confidence, item.until)
    if isinstance(item.place, Window):
        node = index.nodes[item.place.child]
        return WaitingView("silence", node, frozenset(), steps, confidence, item.until)
    pattern, filled = item.place
    return WaitingView("set", index.sets[pattern], filled, steps, confidence, item.until)


def miss_view(miss: Miss, index: Index) -> MissView:
    """View of a miss; its place becomes an id of `index`."""
    steps = () if miss.partial is None else tuple(map(event_view, miss.partial.steps))
    if miss.place is None:
        kind, place, filled = None, None, frozenset()
    elif isinstance(miss.place, Cursor):
        kind, place, filled = "node", index.nodes[miss.place], frozenset()
    elif isinstance(miss.place, Window):
        kind, place, filled = "silence", index.nodes[miss.place.child], frozenset()
    else:
        pattern, filled = miss.place
        kind, place = "set", index.sets[pattern]
    return MissView(miss.reason, kind, place, filled, steps, miss.scenario, miss.need, miss.got)
