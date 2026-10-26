"""HTML из view-моделей: шаблоны Jinja2 и состояние графа для кадра."""

from pathlib import Path
from typing import NamedTuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from rule_engine.application.views import Frame, GraphView, MatchView, NodeView, WaitingView
from rule_engine.interfaces.web.svg import layout

TEMPLATES = Path(__file__).parent / "templates"


class State(NamedTuple):
    """Что подсветить на графе в кадре."""

    active: dict[int, int]  # вершина → сколько совпадений в ней стоит
    passed: set[int]  # вершины, ребро в которые пройдено
    fired: set[int]  # вершины, где сработал сценарий
    sets: dict[int, list[WaitingView]]  # набор `any` → совпадения в нём
    fired_sets: set[int]


EMPTY = State({}, set(), set(), {}, set())


class Paths:
    """Пути от корня до вершин графа: по ним подсвечиваются рёбра и ищется место срабатывания."""

    def __init__(self, graph: GraphView) -> None:
        self.graph = graph
        self.parents: dict[int, int | None] = {}
        self.tokens: dict[int, tuple[str, ...]] = {}  # токены рёбер от корня до вершины
        self.nodes: dict[int, NodeView] = {}

        def walk(node: NodeView, parent: int | None, tokens: tuple[str, ...]) -> None:
            self.parents[node.id] = parent
            self.tokens[node.id] = (*tokens, node.edge.token)
            self.nodes[node.id] = node
            for child in node.children:
                walk(child, node.id, self.tokens[node.id])

        for root in graph.roots:
            walk(root, None, ())

    def up(self, node: int | None) -> set[int]:
        """Вершина и все её предки."""
        found = set()
        while node is not None:
            found.add(node)
            node = self.parents[node]
        return found

    def places(self, match: MatchView) -> tuple[set[int], set[int]]:
        """Вершины и наборы, где мог закончиться шаблон срабатывания: тот же сценарий, те же токены.

        Движок не говорит, каким шаблоном сработал сценарий, поэтому место ищется по шагам.
        """
        tokens = tuple(step.token for step in match.steps)
        nodes = {
            node
            for node, path in self.tokens.items()
            if self.nodes[node].scenario == match.scenario and path == tokens
        }
        sets = {
            pattern.id
            for pattern in self.graph.sets
            if pattern.scenario == match.scenario
            and sorted(slot.token for slot in pattern.slots) == sorted(tokens)
        }
        return nodes, sets

    def state(self, frame: Frame) -> State:
        state = State({}, set(), set(), {}, set())
        for item in frame.waiting:
            if item.kind == "node":
                state.active[item.place] = state.active.get(item.place, 0) + 1
                state.passed.update(self.up(item.place))
            else:
                state.sets.setdefault(item.place, []).append(item)
        for match in frame.matches:
            nodes, sets = self.places(match)
            state.fired.update(nodes)
            for node in nodes:
                state.passed.update(self.up(node))
            state.fired_sets.update(sets)
        return state

    def where(self, item: WaitingView) -> str:
        """Где стоит совпадение: путь по рёбрам или набор `any`."""
        if item.kind == "set":
            pattern = self.graph.sets[item.place]
            return f"{{{', '.join(slot.token for slot in pattern.slots)}}} → {pattern.scenario}"
        return " → ".join(self.tokens[item.place])


class Presenter:
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

    def page(
        self,
        rules_files: list[str],
        stream_files: list[str],
        rules_text: str,
        stream_text: str,
        graph: GraphView | None,
        error: str | None,
    ) -> str:
        return self._env.get_template("page.html").render(
            rules_files=rules_files,
            stream_files=stream_files,
            rules_text=rules_text,
            stream_text=stream_text,
            graph=graph,
            error=error,
        )

    def graph(self, graph: GraphView) -> str:
        return self._env.get_template("graph.html").render(graph=graph)

    def trace(self, graph: GraphView, frames: list[Frame], error: str | None) -> str:
        return self._env.get_template("trace.html").render(graph=graph, frames=frames, error=error)

    def error(self, message: str) -> str:
        return self._env.get_template("error.html").render(message=message)
