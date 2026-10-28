"""HTML из view-моделей: шаблоны Jinja2 и состояние графа для кадра."""

from pathlib import Path
from typing import NamedTuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from rule_engine.application.views import (
    Frame,
    GraphView,
    MatchView,
    MissView,
    NodeView,
    WaitingView,
)
from rule_engine.interfaces.web.svg import layout

TEMPLATES = Path(__file__).parent / "templates"


class State(NamedTuple):
    """Что подсветить на графе в кадре."""

    active: dict[int, int]  # вершина → сколько совпадений в ней стоит
    passed: set[int]  # вершины, ребро в которые пройдено
    fired: set[int]  # вершины, где сработал сценарий
    sets: dict[int, list[WaitingView]]  # набор `any` → совпадения в нём
    fired_sets: set[int]
    missed: set[int]  # вершины, ребро в которые не пустило токен: порог или опоздание
    missed_sets: set[int]  # наборы, которые не пустили токен
    progress: dict[int, dict[int, int]]  # ребро → offset → число кандидатов
    rejected: dict[int, set[int]]  # ребро → позиции отказов


EMPTY = State({}, set(), set(), {}, set(), set(), set(), {}, {})

# причина отказа → метка и класс строки: порог уверенности, время, повтор
REASONS = {
    "score": ("порог токена", "score"),
    "pruned": ("общая уверенность", "score"),
    "confidence": ("общая уверенность", "score"),
    "late": ("задержка", "late"),
    "expired": ("задержка", "late"),
    "cooldown": ("cooldown", "late"),
    "repeat": ("повтор", "repeat"),
}


class Note(NamedTuple):
    """Строка «почему не продвинулось»."""

    tag: str
    css: str
    text: str
    count: int  # столько одинаковых отказов: например, у нескольких копий совпадения


class Notes(NamedTuple):
    main: list[Note]  # отказы токена этого кадра и несработавшие сценарии
    expired: list[Note]  # совпадения, снятые по сроку: к токену кадра не относятся
    skipped: bool  # токен никуда не подошёл и ни в чём не отказан


class Paths:
    """Пути от корня до вершин графа: по ним подсвечиваются рёбра и ищется место срабатывания."""

    def __init__(self, graph: GraphView) -> None:
        self.graph = graph
        self.parents: dict[int, int | None] = {}
        self.tokens: dict[int, tuple[str, ...]] = {}  # токены рёбер от корня до вершины
        self.nodes: dict[int, NodeView] = {}

        def walk(node: NodeView, parent: int | None, tokens: tuple[str, ...]) -> None:
            self.parents[node.id] = parent
            self.tokens[node.id] = (*tokens, *(edge.token for edge in node.segment))
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
        state = State({}, set(), set(), {}, set(), set(), set(), {}, {})
        for item in frame.waiting:
            if item.kind == "node":
                offsets = state.progress.setdefault(item.place, {})
                offsets[item.offset] = offsets.get(item.offset, 0) + 1
                if item.offset == len(self.nodes[item.place].segment):
                    state.active[item.place] = state.active.get(item.place, 0) + 1
                    state.passed.update(self.up(item.place))
                else:
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
                state.rejected.setdefault(miss.place, set()).add(miss.offset)
            elif miss.reason in ("score", "late") and miss.kind == "set":
                state.missed_sets.add(miss.place)
        return state

    def set_name(self, place: int) -> str:
        pattern = self.graph.sets[place]
        return f"{{{', '.join(slot.token for slot in pattern.slots)}}} «{pattern.scenario}»"

    def explain(self, miss: MissView, token: str) -> str:
        """Отказ словами: что не продвинулось, и с чем сравнили."""
        score = f"уверенность {miss.got:.2f} ниже порога {miss.need:g}"
        late = f"ждали до {miss.need:g}, а токен пришёл в {miss.got:g}"
        if miss.reason in ("confidence", "repeat", "cooldown"):
            head = f"«{miss.scenario}» не сработал"
            if miss.reason == "confidence":
                return f"{head}: общая уверенность {miss.got:.2f} ниже порога {miss.need:g}"
            if miss.reason == "repeat":
                return f"{head} повторно: эти токены уже вошли в его срабатывание"
            return f"{head}: после прошлого срабатывания молчит до {miss.need:g}"
        if miss.kind == "set":
            name = self.set_name(miss.place)
            if miss.reason == "expired":
                return f"набор {name} снят: ждал следующий шаг до {miss.need:g}"
            if miss.reason == "pruned":
                return (
                    f"набор {name} отсеян: даже с уверенностью 1.0 у остальных шагов общая "
                    f"будет не выше {miss.got:.2f}, а порог {miss.need:g}"
                )
            if not miss.steps:
                return f"набор {name} не начался с «{token}»: {score}"
            why = late if miss.reason == "late" else score
            return f"набор {name} не закрыл шаг «{token}»: {why}"
        tokens = self.prefix(miss.place, miss.offset)
        path = " → ".join(tokens)
        if miss.reason == "expired":
            return f"«{path}» снято: ждало продолжения до {miss.need:g}"
        if miss.reason == "pruned":
            return (
                f"«{path}» отсеяно: даже с уверенностью 1.0 у остальных шагов общая будет не "
                f"выше {miss.got:.2f}, а порог {miss.need:g}"
            )
        if not miss.steps:
            return f"новое совпадение с «{token}» не началось: {score}"
        before = " → ".join(tokens[:-1])
        why = late if miss.reason == "late" else score
        return f"«{before}» не пошло дальше по «{token}»: {why}"

    def notes(self, frame: Frame) -> Notes:
        main: dict[tuple[str, str, str], int] = {}
        expired: dict[tuple[str, str, str], int] = {}
        for miss in frame.misses:
            tag, css = REASONS[miss.reason]
            key = (tag, css, self.explain(miss, frame.event.token))
            target = expired if miss.reason == "expired" else main
            target[key] = target.get(key, 0) + 1
        position = frame.event.position
        used = any(
            step.position == position
            for item in (*frame.waiting, *frame.matches)
            for step in item.steps
        )
        return Notes(
            [Note(*key, count) for key, count in main.items()],
            [Note(*key, count) for key, count in expired.items()],
            not used and not main,
        )

    def where(self, item: WaitingView) -> str:
        """Где стоит совпадение: путь по рёбрам или набор `any`."""
        if item.kind == "set":
            pattern = self.graph.sets[item.place]
            return f"{{{', '.join(slot.token for slot in pattern.slots)}}} → {pattern.scenario}"
        path = " → ".join(self.prefix(item.place, item.offset))
        return f"{path} (шаг {item.offset}/{len(self.nodes[item.place].segment)} ребра)"

    def prefix(self, place: int, offset: int) -> tuple[str, ...]:
        tokens = self.tokens[place]
        return tokens[:len(tokens) - len(self.nodes[place].segment) + offset]


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
        session: str,
    ) -> str:
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
        return self._env.get_template("graph.html").render(graph=graph)

    def trace(self, graph: GraphView, frames: list[Frame], error: str | None) -> str:
        return self._env.get_template("trace.html").render(graph=graph, frames=frames, error=error)

    def error(self, message: str) -> str:
        return self._env.get_template("error.html").render(message=message)
