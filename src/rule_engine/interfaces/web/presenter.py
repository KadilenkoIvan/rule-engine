"""HTML from view models: Jinja2 templates and the graph state of a frame."""

from pathlib import Path
from typing import NamedTuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from rule_engine.application.tokens import fits, token_label
from rule_engine.application.views import (
    Frame,
    GraphView,
    MatchView,
    MissView,
    NodeView,
    SetView,
    SilenceEdge,
    WaitingView,
)
from rule_engine.interfaces.web.svg import Layout, layout

TEMPLATES = Path(__file__).parent / "templates"


class State(NamedTuple):
    """What to light up on the graph in a frame."""

    active: dict[int, int]  # вершина → сколько совпадений в ней стоит
    passed: set[int]  # вершины, ребро в которые пройдено
    fired: set[int]  # вершины, где сработал сценарий
    sets: dict[int, list[WaitingView]]  # набор `any` → совпадения в нём
    fired_sets: set[int]
    missed: set[int]  # вершины, ребро в которые не пустило токен: порог, опоздание, тишина
    missed_sets: set[int]  # наборы, которые не пустили токен
    hushed: set[int]  # вершины за рёбрами-тишинами, где идёт окно


def empty() -> State:
    """State with nothing lit."""
    return State({}, set(), set(), {}, set(), set(), set(), set())


EMPTY = empty()

# причина отказа → метка и класс строки: порог уверенности, время, повтор
REASONS = {
    "score": ("порог токена", "score"),
    "pruned": ("общая уверенность", "score"),
    "confidence": ("общая уверенность", "score"),
    "late": ("задержка", "late"),
    "expired": ("задержка", "late"),
    "silence": ("тишина", "late"),
    "cooldown": ("cooldown", "late"),
    "repeat": ("повтор", "repeat"),
    "corrected": ("поправка", "marker"),
    "negated": ("отрицание", "marker"),
}


class Note(NamedTuple):
    """Line of «почему не продвинулось»."""

    tag: str
    css: str
    text: str
    count: int  # столько одинаковых отказов: например, у нескольких копий совпадения


class Notes(NamedTuple):
    """Miss lines of a frame."""

    main: list[Note]  # отказы токена этого кадра и несработавшие сценарии
    expired: list[Note]  # совпадения, снятые по сроку: к токену кадра не относятся
    skipped: bool  # токен никуда не подошёл и ни в чём не отказан


def covers(slots: tuple[str, ...], tokens: tuple[str, ...], vocabulary: frozenset[str]) -> bool:
    """Whether the tokens can be dealt to the set steps one each, every token fitting its step.

    Kuhn's matching: `any` fits any token, so comparing sorted lists is not enough.
    """
    if len(slots) != len(tokens):
        return False
    owner: dict[int, int] = {}  # шаг → токен
    return all(
        _seat(token, set(), slots, tokens, vocabulary, owner) for token in range(len(tokens))
    )


def _seat(
    token: int,
    seen: set[int],
    slots: tuple[str, ...],
    tokens: tuple[str, ...],
    vocabulary: frozenset[str],
    owner: dict[int, int],
) -> bool:
    """Augmenting path of `covers`: finds a step for `token`, moving other tokens if needed.

    Args:
        token: index of the token.
        seen: steps already tried on this path; filled here.
        slots: step tokens.
        tokens: event tokens.
        vocabulary: the vocabulary, for `any_vac`.
        owner: step → index of its token; updated here.
    """
    for slot, wanted in enumerate(slots):
        if slot not in seen and fits(wanted, tokens[token], vocabulary):
            seen.add(slot)
            if slot not in owner or _seat(owner[slot], seen, slots, tokens, vocabulary, owner):
                owner[slot] = token
                return True
    return False


def _slots(pattern: SetView) -> str:
    """Steps of an `any` pattern in braces."""
    return f"{{{', '.join(token_label(slot.token) for slot in pattern.slots)}}}"


def _score(miss: MissView) -> str:
    """Score compared with the threshold."""
    return f"уверенность {miss.got:.2f} ниже порога {miss.need:g}"


def _late(miss: MissView) -> str:
    """Arrival time compared with the deadline."""
    return f"ждали до {miss.need:g}, а токен пришёл в {miss.got:g}"


def _explain_scenario(miss: MissView) -> str:
    """Miss of a scenario: low confidence, a repeat or cooldown."""
    head = f"«{miss.scenario}» не сработал"
    if miss.reason == "confidence":
        return f"{head}: общая уверенность {miss.got:.2f} ниже порога {miss.need:g}"
    if miss.reason == "repeat":
        return f"{head} повторно: эти токены уже вошли в его срабатывание"
    return f"{head}: после прошлого срабатывания молчит до {miss.need:g}"


def _used(frame: Frame) -> bool:
    """Whether the frame's event is a step of a match, a held completion or a partial match."""
    position = frame.event.position
    return any(
        step.position == position
        for item in (*frame.waiting, *frame.held, *frame.matches)
        for step in item.steps
    )


class Paths:
    """Paths from the root to graph nodes: they light up edges and locate where matches ended."""

    def __init__(self, graph: GraphView) -> None:
        self.graph = graph
        self.parents: dict[int, int | None] = {}
        self.tokens: dict[int, tuple[str, ...]] = {}  # токены рёбер от корня до вершины, без тишин
        self.labels: dict[int, tuple[str, ...]] = {}  # подписи рёбер от корня до вершины
        self.nodes: dict[int, NodeView] = {}
        for root in graph.roots:
            self._walk(root, None)

    def _walk(self, node: NodeView, parent: int | None) -> None:
        """Records the path to `node` and to every node below it."""
        self.parents[node.id] = parent
        tokens = () if parent is None else self.tokens[parent]
        labels = () if parent is None else self.labels[parent]
        if isinstance(node.edge, SilenceEdge):
            self.tokens[node.id] = tokens
            self.labels[node.id] = (*labels, f"тишина {node.edge.duration:g} с")
        else:
            self.tokens[node.id] = (*tokens, node.edge.token)
            self.labels[node.id] = (*labels, token_label(node.edge.token))
        self.nodes[node.id] = node
        for child in node.children:
            self._walk(child, node.id)

    def up(self, node: int | None) -> set[int]:
        """The node and all its ancestors."""
        found = set()
        while node is not None:
            found.add(node)
            node = self.parents[node]
        return found

    def places(self, match: MatchView) -> tuple[set[int], set[int]]:
        """Nodes and sets where a pattern of the match could end.

        The engine does not say which pattern fired, so places are found by the steps: the same
        scenario, and the tokens fit the steps.

        Returns:
            Node ids and set ids.
        """
        tokens = tuple(step.token for step in match.steps)
        vocabulary = frozenset(self.graph.vocabulary)
        nodes = {
            node
            for node, path in self.tokens.items()
            if self.nodes[node].scenario == match.scenario
            and len(path) == len(tokens)
            and all(fits(wanted, token, vocabulary) for wanted, token in zip(path, tokens))
        }
        sets = {
            pattern.id
            for pattern in self.graph.sets
            if pattern.scenario == match.scenario
            and covers(tuple(slot.token for slot in pattern.slots), tokens, vocabulary)
        }
        return nodes, sets

    def state(self, frame: Frame) -> State:
        """What to light up on the graph in the frame."""
        state = empty()
        for item in frame.waiting:
            if item.kind == "node":
                state.active[item.place] = state.active.get(item.place, 0) + 1
                state.passed.update(self.up(item.place))
            elif item.kind == "silence":
                state.hushed.add(item.place)
                state.passed.update(self.up(self.parents[item.place]))
            else:
                state.sets.setdefault(item.place, []).append(item)
        for match in frame.matches:
            nodes, sets = self.places(match)
            state.fired.update(nodes)
            for node in nodes:
                state.passed.update(self.up(node))
            state.fired_sets.update(sets)
        for miss in frame.misses:
            if miss.reason in ("score", "late") and miss.kind == "node":
                state.missed.add(miss.place)
            elif miss.reason == "silence":
                state.missed.add(miss.place)
            elif miss.reason in ("score", "late") and miss.kind == "set":
                state.missed_sets.add(miss.place)
        return state

    def set_name(self, place: int) -> str:
        """Name of an `any` set: its steps and scenario."""
        pattern = self.graph.sets[place]
        return f"{_slots(pattern)} «{pattern.scenario}»"

    def explain(self, miss: MissView, token: str | None) -> str:
        """Miss in words: what did not advance and what it was compared with.

        Args:
            miss: the miss.
            token: token of the frame's event; None for a timer frame.
        """
        if miss.reason in ("corrected", "negated"):
            return self._explain_marker(miss)
        if miss.reason in ("confidence", "repeat", "cooldown"):
            return _explain_scenario(miss)
        if miss.kind == "set":
            return self._explain_set(miss, token)
        if miss.reason == "silence":
            return self._explain_silence(miss, token)
        return self._explain_node(miss, token)

    def _explain_marker(self, miss: MissView) -> str:
        """Miss of the correction mode: removed by a correction or suppressed by a negation."""
        if miss.reason == "corrected" and miss.scenario is not None:
            return f"«{miss.scenario}» снят поправкой в {miss.got:g}: ждал выдачи до {miss.need:g}"
        if miss.scenario is not None:
            return f"«{miss.scenario}» подавлен отрицанием, открытым в {miss.need:g}"
        where = (
            f"набор {self.set_name(miss.place)}"
            if miss.kind == "set"
            else f"«{' → '.join(self.labels[miss.place])}»"
        )
        what = "снятого" if miss.reason == "corrected" else "подавленного"
        return f"{where} снято: в нём последний токен {what} срабатывания"

    def _explain_set(self, miss: MissView, token: str | None) -> str:
        """Miss of an `any` set."""
        name = self.set_name(miss.place)
        if miss.reason == "expired":
            return f"набор {name} снят: ждал следующий шаг до {miss.need:g}"
        if miss.reason == "pruned":
            return (
                f"набор {name} отсеян: даже с уверенностью 1.0 у остальных шагов общая "
                f"будет не выше {miss.got:.2f}, а порог {miss.need:g}"
            )
        if not miss.steps:
            return f"набор {name} не начался с «{token}»: {_score(miss)}"
        why = _late(miss) if miss.reason == "late" else _score(miss)
        return f"набор {name} не закрыл шаг «{token}»: {why}"

    def _explain_silence(self, miss: MissView, token: str | None) -> str:
        """Broken silence: before the step or inside the window."""
        before = " → ".join(self.labels[miss.place][:-1])
        last = miss.steps[-1].time
        if miss.got < last:
            return (
                f"«{before}» не начало ждать тишину: запрещённый токен пришёл в {miss.got:g}, "
                f"за {last - miss.got:.1f} с до шага"
            )
        return (
            f"«{before}» не дождалось тишины: «{token}» пришёл в {miss.got:g}, а тишина "
            f"нужна до {miss.need:g}"
        )

    def _explain_node(self, miss: MissView, token: str | None) -> str:
        """Miss of a trie position."""
        path = " → ".join(self.labels[miss.place])
        if miss.reason == "expired":
            return f"«{path}» снято: ждало продолжения до {miss.need:g}"
        if miss.reason == "pruned":
            return (
                f"«{path}» отсеяно: даже с уверенностью 1.0 у остальных шагов общая будет не "
                f"выше {miss.got:.2f}, а порог {miss.need:g}"
            )
        if not miss.steps:
            return f"новое совпадение с «{token}» не началось: {_score(miss)}"
        before = " → ".join(self.labels[miss.place][:-1])
        why = _late(miss) if miss.reason == "late" else _score(miss)
        return f"«{before}» не пошло дальше по «{token}»: {why}"

    def notes(self, frame: Frame) -> Notes:
        """Lines «почему не продвинулось» of the frame; equal misses are counted, not repeated."""
        main: dict[tuple[str, str, str], int] = {}
        expired: dict[tuple[str, str, str], int] = {}
        token = None if frame.event is None else frame.event.token
        for miss in frame.misses:
            tag, css = REASONS[miss.reason]
            key = (tag, css, self.explain(miss, token))
            target = expired if miss.reason == "expired" else main
            target[key] = target.get(key, 0) + 1
        # у кадра по таймеру пропускать нечего
        skipped = frame.event is not None and not main and not _used(frame)
        return Notes(
            [Note(*key, count) for key, count in main.items()],
            [Note(*key, count) for key, count in expired.items()],
            skipped,
        )

    def where(self, item: WaitingView) -> str:
        """Where the match stands: the path of edges or the `any` set."""
        if item.kind == "set":
            pattern = self.graph.sets[item.place]
            return f"{_slots(pattern)} → {pattern.scenario}"
        return " → ".join(self.labels[item.place])


class Board(NamedTuple):
    """Graph of a live session with its layout and paths: built once per session."""

    graph: GraphView
    layout: Layout
    paths: Paths


class Presenter:
    """Renders pages and fragments of the web interface."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(TEMPLATES),
            autoescape=True,  # в токенах есть < и >
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        self._env.filters["g"] = lambda value: f"{value:g}"
        self._env.filters["score"] = lambda value: f"{value:.2f}"
        self._env.globals["layout"] = layout
        self._env.globals["EMPTY"] = EMPTY
        self._env.globals["Paths"] = Paths
        # макросы живого режима зовутся по одному: кадр приходит, когда его отдал движок
        self._macros = self._env.get_template("macros.html").module
        self._live = self._env.get_template("live.html").module

    def page(
        self,
        rules_files: list[str],
        stream_files: list[str],
        rules_text: str,
        stream_text: str,
        graph: GraphView | None,
        error: str | None,
        session: str,
    ) -> str:
        """Main page: example lists, rule and stream fields, the graph or the rules error."""
        return self._env.get_template("page.html").render(
            rules_files=rules_files,
            stream_files=stream_files,
            rules_text=rules_text,
            stream_text=stream_text,
            graph=graph,
            error=error,
            session=session,
        )

    def graph(self, graph: GraphView) -> str:
        """Graph fragment."""
        return self._env.get_template("graph.html").render(graph=graph)

    def trace(self, graph: GraphView, frames: list[Frame], error: str | None) -> str:
        """Stream run fragment: a player over the frames; `error` stops the stream."""
        return self._env.get_template("trace.html").render(graph=graph, frames=frames, error=error)

    def error(self, message: str) -> str:
        """Error fragment."""
        return self._env.get_template("error.html").render(message=message)

    def board(self, graph: GraphView) -> Board:
        """Graph of a live session with its layout and paths."""
        return Board(graph, layout(graph), Paths(graph))

    def live_start(self, board: Board) -> tuple[str, str]:
        """Start of a live session: vocabulary token buttons and a player with an empty graph."""
        return str(self._live.tokens(board.graph)), str(self._live.view(board.graph, board.layout))

    def live_frame(self, board: Board, frame: Frame, index: int) -> tuple[str, str]:
        """Frame of a live session and its lines for the match feed."""
        html = self._macros.frame_view(board.graph, board.layout, board.paths, frame, index, False)
        return str(html), str(self._live.log(frame, index))
