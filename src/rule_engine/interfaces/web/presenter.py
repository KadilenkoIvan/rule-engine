"""HTML из view-моделей: шаблоны Jinja2 и состояние графа для кадра."""

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
    SilenceEdge,
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
    missed: set[int]  # вершины, ребро в которые не пустило токен: порог, опоздание, тишина
    missed_sets: set[int]  # наборы, которые не пустили токен
    hushed: set[int]  # вершины за рёбрами-тишинами, где идёт окно


def empty() -> State:
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


def covers(slots: tuple[str, ...], tokens: tuple[str, ...], vocabulary: frozenset[str]) -> bool:
    """Можно ли раздать токены шагам набора по одному, чтобы каждый подошёл к своему шагу.

    Паросочетание Куна: `any` подходит к любому токену, поэтому простого сравнения
    отсортированных списков мало.
    """
    if len(slots) != len(tokens):
        return False
    owner: dict[int, int] = {}  # шаг → токен

    def place(token: int, seen: set[int]) -> bool:
        for slot, wanted in enumerate(slots):
            if slot not in seen and fits(wanted, tokens[token], vocabulary):
                seen.add(slot)
                if slot not in owner or place(owner[slot], seen):
                    owner[slot] = token
                    return True
        return False

    return all(place(token, set()) for token in range(len(tokens)))


class Paths:
    """Пути от корня до вершин графа: по ним подсвечиваются рёбра и ищется место срабатывания."""

    def __init__(self, graph: GraphView) -> None:
        self.graph = graph
        self.parents: dict[int, int | None] = {}
        self.tokens: dict[int, tuple[str, ...]] = {}  # токены рёбер от корня до вершины, без тишин
        self.labels: dict[int, tuple[str, ...]] = {}  # подписи рёбер от корня до вершины
        self.nodes: dict[int, NodeView] = {}

        def walk(node: NodeView, parent: int | None) -> None:
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
                walk(child, node.id)

        for root in graph.roots:
            walk(root, None)

    def up(self, node: int | None) -> set[int]:
        """Вершина и все её предки."""
        found = set()
        while node is not None:
            found.add(node)
            node = self.parents[node]
        return found

    def places(self, match: MatchView) -> tuple[set[int], set[int]]:
        """Вершины и наборы, где мог закончиться шаблон срабатывания: тот же сценарий, токены
        подходят к шагам (`any` — к любому).

        Движок не говорит, каким шаблоном сработал сценарий, поэтому место ищется по шагам.
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
        pattern = self.graph.sets[place]
        return f"{{{', '.join(token_label(slot.token) for slot in pattern.slots)}}} «{pattern.scenario}»"

    def explain(self, miss: MissView, token: str | None) -> str:
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
        path = " → ".join(self.labels[miss.place])
        if miss.reason == "silence":
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
        if miss.reason == "expired":
            return f"«{path}» снято: ждало продолжения до {miss.need:g}"
        if miss.reason == "pruned":
            return (
                f"«{path}» отсеяно: даже с уверенностью 1.0 у остальных шагов общая будет не "
                f"выше {miss.got:.2f}, а порог {miss.need:g}"
            )
        if not miss.steps:
            return f"новое совпадение с «{token}» не началось: {score}"
        before = " → ".join(self.labels[miss.place][:-1])
        why = late if miss.reason == "late" else score
        return f"«{before}» не пошло дальше по «{token}»: {why}"

    def notes(self, frame: Frame) -> Notes:
        main: dict[tuple[str, str, str], int] = {}
        expired: dict[tuple[str, str, str], int] = {}
        token = None if frame.event is None else frame.event.token
        for miss in frame.misses:
            tag, css = REASONS[miss.reason]
            key = (tag, css, self.explain(miss, token))
            target = expired if miss.reason == "expired" else main
            target[key] = target.get(key, 0) + 1
        if frame.event is None:
            # кадр по таймеру: пропускать нечего
            return Notes(
                [Note(*key, count) for key, count in main.items()],
                [Note(*key, count) for key, count in expired.items()],
                False,
            )
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
            return f"{{{', '.join(token_label(slot.token) for slot in pattern.slots)}}} → {pattern.scenario}"
        return " → ".join(self.labels[item.place])


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
