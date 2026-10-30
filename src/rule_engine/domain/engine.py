"""Matching of a token stream against the scenario trie, `any` patterns and silence windows.

An event touches only what waits for it: token indexes find trie positions, `any` sets and
broken silence windows, and the deadline queue finds expired candidates. Candidates of one place
live in a `Front` with the best continuation and the earliest deadlines at its ends.
"""

import heapq
import math
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import chain, count, groupby, islice, takewhile
from typing import NamedTuple

from rule_engine.domain.graph import (
    Cursor,
    Edge,
    Node,
    Silence,
    Start,
    build_graph,
    insert_scenario,
    longest_before,
)
from rule_engine.domain.rules import (
    WILDCARDS,
    PatternStep,
    Phrase,
    RuleSet,
    Scenario,
    SilenceStep,
    fits,
    fitting,
    parse_scenario,
    step_tokens,
)
from rule_engine.domain.timer_wheel import Timer, TimerWheel
from rule_engine.domain.unordered import Unordered, build_unordered, insert_unordered


@dataclass(frozen=True)
class Event:
    """Token from the detector."""

    token: str
    score: float  # уверенность детектора: шаг принимает токен, только если она не ниже порога
    time: float  # секунды: когда токен сказан, по одной шкале для всего потока (ставит детектор)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"уверенность {self.score} токена {self.token!r} вне [0, 1]")
        if not math.isfinite(self.time):
            raise ValueError(f"время {self.time} токена {self.token!r} не конечное число")


class Step(NamedTuple):
    """Event of the stream with its position."""

    position: int  # сквозной номер события в потоке, между порциями не сбрасывается
    event: Event


@dataclass(frozen=True)
class Match:
    """Fired scenario."""

    scenario: str
    steps: tuple[Step, ...]  # шаги с токенами; шаги-тишины в срабатывание не входят
    confidence: float  # общая уверенность: среднее геометрическое score шагов
    # режим поправок: срок, после которого выдано отложенное срабатывание; None — выдано сразу
    released: float | None = None


@dataclass(frozen=True, slots=True)
class History:
    """Immutable history link: candidates share common prefixes."""

    step: Step
    previous: "History | None" = None


class Partial(NamedTuple):
    """Partial match: step history and the time continuations are measured from."""

    tail: History
    product: float
    count: int
    first_position: int
    last_time: float

    @classmethod
    def begin(cls, step: Step) -> "Partial":
        """Match of one step."""
        return cls(History(step), step.event.score, 1, step.position, step.event.time)

    def backwards(self) -> Iterator[Step]:
        """Steps from the last one to the first."""
        link = self.tail
        while link is not None:
            yield link.step
            link = link.previous

    @property
    def steps(self) -> tuple[Step, ...]:
        """Steps in order; built only for results and views."""
        steps = list(self.backwards())
        steps.reverse()
        return tuple(steps)

    def positions(self) -> list[int]:
        """Step positions in order: a tie-break when confidence and start are equal."""
        positions = [step.position for step in self.backwards()]
        positions.reverse()
        return positions

    @property
    def confidence(self) -> float:
        """Geometric mean of step scores."""
        return self.product ** (1 / self.count)

    def bound(self, length: int) -> float:
        """Best confidence reachable in a pattern of `length` steps: the rest scored 1.0."""
        return self.product ** (1 / length)

    def extend(self, step: Step) -> "Partial":
        """Match continued by `step`."""
        return Partial(
            History(step, self.tail),
            self.product * step.event.score,
            self.count + 1,
            self.first_position,
            step.event.time,
        )


class Window(NamedTuple):
    """Silence window: a trie node and the silence edge out of it."""

    node: Node
    edge: Silence

    @property
    def child(self) -> Cursor:
        """Where the match goes if the window passes without forbidden tokens."""
        return Cursor(self.node.silences[self.edge], 1)


class Hush(NamedTuple):
    """Timer payload: a match waiting for the window to pass without forbidden tokens."""

    window: Window
    partial: Partial


# Где стоит совпадение: курсор radix trie, набор `any` или окно тишины.
Place = Cursor | tuple[Unordered, frozenset[int]] | Window


class Waiting(NamedTuple):
    """Partial match in an engine snapshot."""

    place: Place
    partial: Partial
    until: float  # секунды: срок продолжения, а у окна тишины — её конец


class Held(NamedTuple):
    """Completed match in correction mode: the scenario fired, its release waits."""

    scenario: str
    partial: Partial
    until: float  # секунды: выдаётся, когда время станет больше
    frozen: bool  # ждёт решения отложенного маркера: по сроку не выдаётся


class Marker:
    """Marker phrase whose token is a step of a live match: its action waits.

    A fired match with that step cancels the marker. Once all such matches die, the marker is
    applied as it would have been at its own moment.
    """

    __slots__ = ("kind", "steps", "time", "state", "holds")

    def __init__(self, kind: str, steps: list[Step], time: float) -> None:
        self.kind = kind  # "correction", "negation" или "break"
        self.steps = steps  # шаги фразы с токенами из шаблонов: те самые объекты, см. `has`
        self.time = time
        self.state: str | None = None  # None — ждёт; "applied" или "cancelled"
        self.holds: list[Hold] = []  # у поправки: ждущие на её момент


class Negation:
    """Open negation: suppresses the first completed match with a step after it."""

    __slots__ = ("position", "time", "deadline", "marker", "breaks")

    def __init__(self, position: int, time: float, deadline: float, marker: Marker | None) -> None:
        self.position = position  # последний токен фразы отрицания
        self.time = time
        self.deadline = deadline  # закрывается, когда время станет больше
        self.marker = marker  # отложенная фраза отрицания; None — отрицание открыто точно
        self.breaks: list[Marker] = []  # отложенные разрывы и поправки: применятся — закроют


class Hold:
    """Held completion: new steps of matches with the same first step extend its deadline."""

    __slots__ = ("scenario", "partial", "time", "hold", "deadline", "waits")

    def __init__(self, scenario: str, partial: Partial, time: float, hold: float) -> None:
        self.scenario = scenario
        self.partial = partial
        self.time = time  # момент завершения: от него считался cooldown
        self.hold = hold
        self.deadline = time + hold
        self.waits: set[Marker | Victims] = set()  # чьего решения ждёт выдача


class Victims:
    """Completions of one moment that a tentative negation would suppress.

    They are suppressed if the negation marker is applied and its breaks are cancelled.
    """

    __slots__ = ("marker", "breaks", "time", "holds")

    def __init__(self, negation: Negation, holds: list[Hold]) -> None:
        self.marker = negation.marker
        self.breaks = list(negation.breaks)
        self.time = negation.time
        self.holds = holds


class Miss(NamedTuple):
    """Why an event did not advance a match or let a scenario fire.

    `need` is what was required and `got` is what came, by `reason`:

    - "score": step threshold, token score; "pruned": threshold, best reachable confidence;
    - "late", "expired": match deadline, event time; "silence": window end, forbidden token time
      (earlier than the last step if the token came before it);
    - "confidence": scenario threshold, match confidence; "cooldown": cooldown end, time;
      "repeat": 0 and 0, a token is already in a released match of the scenario;
    - "corrected": held deadline, correction time; "negated": negation time, completion time.
      For a partial match removed with them, both are the removal time.

    Scenario misses have `scenario` and no `place`; `partial` is None if no match started.
    """

    reason: str
    place: Place | None
    partial: Partial | None
    scenario: str | None
    need: float
    got: float


def has(partial: Partial, steps: list[Step]) -> bool:
    """Whether the match contains one of `steps`: the same object, not an equal one.

    An explicit token step and an `any` step of one event are different `Step` objects.
    """
    low = min(step.position for step in steps)
    for step in takewhile(lambda item: item.position >= low, partial.backwards()):
        if any(step is mine for mine in steps):
            return True
    return False


def pattern_tokens(scenarios: Iterable[Scenario]) -> set[str]:
    """Tokens of pattern steps, without silence tokens."""
    return {
        step.token
        for scenario in scenarios
        for pattern in scenario.patterns
        for step in pattern
        if isinstance(step, PatternStep)
    }


def later(a: Partial, b: Partial) -> bool:
    """Whether the steps of `a` are not earlier than those of `b`, compared left to right."""
    if a.first_position != b.first_position:
        return a.first_position > b.first_position
    return a.positions() >= b.positions()


def dominates(a: Partial, b: Partial) -> bool:
    """Whether `a` evicts `b` from a place.

    `a` continues not earlier (by continuation time, then by last step) with a product not
    lower, so every continuation of `b` is also one of `a`. On a full tie the later steps win
    (`later`): the result does not depend on the order of arrival.
    """
    if a.last_time != b.last_time:
        return a.last_time > b.last_time and a.product >= b.product
    first, second = a.tail.step.position, b.tail.step.position
    if first != second:
        return first > second and a.product >= b.product
    if a.product != b.product:
        return a.product > b.product
    return later(a, b)


def mark(partial: Partial) -> tuple[float, int]:
    """Order in a front: continuation time, then last step."""
    return partial.last_time, partial.tail.step.position


EMPTY: dict = {}  # переходы фронта набора `any`: их нет, словарь не заводится


class Front:
    """Incomparable matches of one place: a trie position or an `any` set state.

    Matches go by increasing `mark` with strictly decreasing products. So a new match evicts
    only the tail, deadlines expire at the head, and the most confident match within an edge
    gap is the first of the fitting tail.
    """

    __slots__ = ("place", "items", "head", "edges", "queued")

    def __init__(self, place: "Cursor | tuple[Collector, int]") -> None:
        self.place = place
        self.items: list[Partial | None] = []  # до `head` — None: снятые не держат историю
        self.head = 0  # совпадения до `head` сняты по сроку
        # у позиции бора: токен перехода → переходы с ним; по этим токенам фронт стоит в индексе
        self.edges: dict[str, list[Edge]] = {} if isinstance(place, Cursor) else EMPTY
        # есть запись в очереди сроков; поправка может опустошить фронт, а запись останется
        self.queued = False

    def __len__(self) -> int:
        return len(self.items) - self.head

    def __iter__(self) -> Iterator[Partial]:
        return islice(self.items, self.head, None)

    def first(self) -> Partial:
        """Head match: the earliest deadline."""
        return self.items[self.head]

    def add(self, partial: Partial) -> int:
        """Appends a match whose last step is not earlier than the others'.

        Only the last match can evict the new one: only it may share the last step.

        Returns:
            Change of the front size.
        """
        items, head = self.items, self.head
        if len(items) > head and dominates(items[-1], partial):
            return 0
        size = len(items)
        while len(items) > head and dominates(partial, items[-1]):
            items.pop()
        items.append(partial)
        return len(items) - size

    def insert(self, partial: Partial) -> int:
        """Puts a match at its place by `mark`.

        Only the first match not earlier than the new one can evict it: it has the largest
        product among them. The new match evicts the matches right before it and those equal
        to it by `mark`.

        Returns:
            Change of the front size.
        """
        items, head = self.items, self.head
        key = mark(partial)
        index = bisect_left(items, key, head, key=mark)
        if index < len(items) and dominates(items[index], partial):
            return 0
        size = len(items)
        end = index
        while end < len(items) and mark(items[end]) == key and dominates(partial, items[end]):
            end += 1
        start = index
        while start > head and dominates(partial, items[start - 1]):
            start -= 1
        items[start:end] = [partial]
        return len(items) - size

    def pop_first(self) -> Partial:
        """Removes and returns the head match."""
        partial = self.items[self.head]
        self.items[self.head] = None
        self.head += 1
        if self.head * 2 >= len(self.items):  # сжатие: в среднем O(1) на снятое совпадение
            del self.items[: self.head]
            self.head = 0
        return partial

    def since(self, fresh: Callable[[Partial], bool]) -> int:
        """Index of the first match for which `fresh` holds; it holds for all after it."""
        return bisect_left(self.items, True, self.head, key=fresh)


class Collector:
    """`any` pattern in the engine: closed steps are a bit mask, equal steps form one group."""

    __slots__ = ("pattern", "groups", "full", "states")

    def __init__(self, pattern: Unordered) -> None:
        self.pattern = pattern
        self.groups = pattern.groups()  # токен шага → (порог, маска номеров) групп
        self.full = (1 << len(pattern.slots)) - 1  # маска собранного шаблона
        self.states: dict[int, Front] = {}  # маска закрытых шагов → совпадения, без пустых

    def fitting(self, token: str, vocabulary: frozenset[str]) -> list[tuple[float, int]]:
        """Step groups the event token fits: the same for all states of the set."""
        return [group for key in fitting(token, vocabulary) for group in self.groups.get(key, ())]

    def closes(
        self, filled: int, groups: list[tuple[float, int]], score: float
    ) -> tuple[list[int], list[int]]:
        """`Unordered.free` on masks.

        Returns:
            Steps the event closes, and open steps with its token whose threshold it missed.
        """
        free, low = [], []
        for min_score, mask in groups:
            open_ = mask & ~filled
            if open_:
                index = (open_ & -open_).bit_length() - 1
                (free if score >= min_score else low).append(index)
        return free, low

    def place(self, filled: int) -> tuple[Unordered, frozenset[int]]:
        """State as snapshots and misses show it: the pattern and indexes of closed steps."""
        slots = range(len(self.pattern.slots))
        return self.pattern, frozenset(index for index in slots if filled >> index & 1)


def before_of(scenario: Scenario) -> float:
    """Longest backward silence window of the scenario; 0 if no silence looks back."""
    return max(
        (
            step.before
            for pattern in scenario.patterns
            for step in pattern
            if isinstance(step, SilenceStep)
        ),
        default=0.0,
    )


def due_time(last: float, gap: float) -> float:
    """Queue deadline a few ulp earlier than `last + gap`.

    Expiry checks `now - last > gap`, and the subtraction rounds on its own: a match must leave
    the queue no later than it expires. One that leaves earlier returns to the queue.
    """
    return last + gap - 4 * (math.ulp(last) + math.ulp(gap))


def expiry(last: float, gap: float, sequence: bool) -> float:
    """Last moment a match is still alive: it is removed once time is greater.

    Args:
        last: continuation time of the match.
        gap: gap of its place.
        sequence: True for a trie position (`now - last > gap` removes), False for an `any`
            set (`now > last + gap` removes).
    """
    deadline = last + gap
    if sequence:
        while deadline - last > gap:
            deadline = math.nextafter(deadline, -math.inf)
        while math.nextafter(deadline, math.inf) - last <= gap:
            deadline = math.nextafter(deadline, math.inf)
    return deadline


def best(items: list[Partial], start: int, end: int, step: Step) -> Partial:
    """Best continuation of `items[start:end]` by `step`: the only one left in the next place.

    The continuations share the last step, so the product decides, then `later`. Products in
    the front tail decrease, so the ones equal to the best stand right after it.
    """
    score = step.event.score
    winner = items[start].extend(step)
    for index in range(start + 1, end):
        if items[index].product * score != winner.product:
            break
        partial = items[index].extend(step)
        if later(partial, winner):
            winner = partial
    return winner


def _gap(place: "Cursor | tuple[Collector, int]") -> float:
    """Gap of a front place: of the trie position or of the `any` pattern."""
    return place.max_gap if isinstance(place, Cursor) else place[0].pattern.max_gap


def _shown(place: "Cursor | tuple[Collector, int]") -> Place:
    """Front place as snapshots and misses show it."""
    return place if isinstance(place, Cursor) else place[0].place(place[1])


def _pick(pattern: Unordered, index: int, step: Step, wild: Step) -> Step:
    """The event's step for slot `index`: `step` for its own token, `wild` for a wildcard."""
    return step if pattern.slots[index].token == step.event.token else wild


def _most_confident(partials: list[Partial]) -> Partial:
    """The most confident match; on equal confidence, the one with later steps (`later`)."""
    winner = partials[0]
    for partial in partials[1:]:
        if partial.confidence > winner.confidence or (
            partial.confidence == winner.confidence and later(partial, winner)
        ):
            winner = partial
    return winner


def _relocate(place: Place | None, old: Node, prefix: Node, cut: int) -> Place | None:
    """Place after node `old` was split: its first `cut` steps moved to `prefix`."""
    if isinstance(place, Cursor) and place.node is old:
        return (
            Cursor(prefix, place.offset) if place.offset <= cut else Cursor(old, place.offset - cut)
        )
    return place


def _marker_kinds(rules: RuleSet) -> dict[Phrase, str]:
    """Marker phrase → what it does: "correction", "negation" or "break"."""
    kinds: dict[Phrase, str] = {}
    for kind, phrases in (
        ("correction", rules.corrections),
        ("negation", rules.negations),
        ("break", rules.negation_breaks),
    ):
        kinds.update(dict.fromkeys(phrases, kind))
    return kinds


def _prefixes(partials: list[Partial]) -> dict[int, set[tuple[int, ...]]]:
    """First position → step positions of proper prefixes of the removed completions."""
    prefixes: dict[int, set[tuple[int, ...]]] = {}
    for partial in partials:
        positions = partial.positions()
        prefixes.setdefault(partial.first_position, set()).update(
            tuple(positions[:length]) for length in range(1, len(positions))
        )
    return prefixes


def _shares(partial: Partial, lasts: set[int], low: int) -> bool:
    """Whether the match contains one of the positions `lasts`, none of which is below `low`."""
    steps = takewhile(lambda step: step.position >= low, partial.backwards())
    return any(step.position in lasts for step in steps)


def _copies(partial: Partial, prefixes: dict[int, set[tuple[int, ...]]]) -> bool:
    """Whether the match is a prefix of a removed completion (`_prefixes`)."""
    own = prefixes.get(partial.first_position)
    return own is not None and tuple(partial.positions()) in own


class Engine:
    """Streaming matcher of scenarios: partial matches persist between calls.

    A token that does not continue a match is skipped. A match goes along an edge if the token
    score reaches the edge threshold within the edge gap, and leaves a copy for other
    continuations. A scenario fires when a pattern ends, if the confidence (geometric mean of
    scores) reaches its threshold, no token is in a released match of it and its cooldown is
    over. In correction mode a scenario with `hold` > 0 is released after the hold: a
    correction cancels held matches, a negation suppresses the next one.

    Time of events and `advance` must not decrease; snapshot and miss order is not guaranteed.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._defaults = rules  # стандартные задержка и пороги не меняются
        self._correcting = rules.self_correction  # режим поправок
        self._init_correction(rules)
        self._vocabulary = rules.vocabulary
        self._cooldowns = {scenario.name: scenario.cooldown for scenario in rules.scenarios}
        # сценарий → номер объявления: срабатывания одного момента с одним началом идут так
        self._declared = {scenario.name: index for index, scenario in enumerate(rules.scenarios)}
        self._min_confidences = {
            scenario.name: rules.confidence_threshold(scenario) for scenario in rules.scenarios
        }
        self._init_graph(rules)
        self._fronts: dict[Cursor, Front] = {}  # позиция radix trie → кандидаты, без пустых
        # токен перехода → фронты позиций с таким переходом: событие продолжает только их
        self._waiting: dict[str, dict[Front, None]] = {}
        # (срок головы фронта, номер, фронт): срок бывает раньше настоящего — тогда фронт просто
        # вернётся в очередь, — но не позже. У каждого непустого фронта одна запись
        self._deadlines: list[tuple[float, int, Front]] = []
        self._order = count()  # разводит равные сроки, чтобы куча не сравнивала фронты
        self._wheel: TimerWheel[Hush] = TimerWheel()  # сроки окон тишины
        self._hushes: dict[Timer[Hush], None] = {}  # открытые окна в порядке открытия
        self._windows: dict[str, dict[Timer[Hush], None]] = {}  # запрещённый токен → окна с ним
        self._active_count = 0  # кандидаты во фронтах, открытые окна и ждущие выдачи
        self._lookback = longest_before(self._starts)  # секунды: самое длинное окно назад
        self._recent: deque[Step] = deque()  # события не старше `_lookback`: для тишины до шага
        self._taken: dict[str, set[int]] = {}  # сценарий → позиции токенов его выданных срабатываний
        self._taken_count = 0  # сколько позиций во всех `_taken`
        self._kept = 0  # сколько позиций осталось после последней чистки `_taken`
        self._idle = 0  # вызовы `push` и `advance` с последней чистки `_taken`
        self._fired_at: dict[str, float] = {}  # сценарий → время его последнего выданного срабатывания
        self._position = 0
        self._last_time: float | None = None  # None, пока время не задано событием или advance
        self._misses: list[Miss] = []  # отказы последнего события или advance

    def _init_correction(self, rules: RuleSet) -> None:
        """State of the correction mode: held completions and markers."""
        # сценарий → задержка выдачи; без режима поправок у всех 0: срабатывание выдаётся сразу
        self._holds = {scenario.name: self._hold_of(scenario) for scenario in rules.scenarios}
        self._held: list[Hold] = []  # ждущие срабатывания в порядке завершения
        # сценарий → момент завершения последнего выданного срабатывания: к нему возвращается
        # cooldown, если поправка снимет ждущие
        self._settled_at: dict[str, float] = {}
        # фраза-маркер → что она делает; без режима поправок маркеров нет
        self._phrases: dict[Phrase, str] = _marker_kinds(rules) if self._correcting else {}
        self._lengths = sorted({len(phrase) for phrase in self._phrases})
        # последние события: фраза ищется сравнением хвоста потока, её токены идут подряд
        self._said: deque[Step] = deque(maxlen=max(self._lengths, default=1))
        self._negation: Negation | None = None
        self._marker_tokens = {token for phrase in self._phrases for token in phrase}
        # токены маркеров, которые стоят шагами шаблонов: такой маркер может ждать решения
        self._intercepting = self._marker_tokens & pattern_tokens(rules.scenarios)
        self._deferred: list[Marker] = []  # маркеры, ждущие решения, по порядку
        self._groups: list[Victims] = []  # срабатывания, которые подавит отложенное отрицание
        self._used: list[Step] = []  # явные шаги последних событий в сработавших совпадениях
        # первые позиции совпадений, для которых событие стало шагом; None — вне режима поправок
        self._stepped: set[int] | None = None

    def _init_graph(self, rules: RuleSet) -> None:
        """The trie, the `any` patterns and their token indexes."""
        self._starts = build_graph(rules)
        self._start_index: dict[str, list[Start]] = {}  # токен ребра из корня → эти рёбра
        for start in self._starts:
            self._start_index.setdefault(start.token, []).append(start)
        self._unordered = build_unordered(rules)
        self._unordered_owners = {
            (tuple(sorted(pattern.slots)), pattern.max_gap): pattern.scenario
            for pattern in self._unordered
        }
        self._collectors: list[Collector] = []
        self._collector_index: dict[str, list[Collector]] = {}  # токен шага → наборы с ним
        self._add_collectors(self._unordered)

    def add_scenario(self, scenario: Scenario | dict) -> None:
        """Adds a scenario without rebuilding the graph or dropping current matches.

        Only the new patterns are checked; their tokens join the vocabulary after a successful
        insert. Kept prefixes serve the new scenario; events are not replayed. Not thread-safe:
        call between `push`/`feed`.

        Args:
            scenario: `Scenario` or a dict in the format of a `scenarios` item of the JSON.

        Raises:
            ValueError: the name is taken, or the scenario is invalid or conflicts with others.
        """
        if isinstance(scenario, dict):
            scenario = parse_scenario(scenario)
        if scenario.name in self._cooldowns:
            raise ValueError(f"сценарий {scenario.name!r} объявлен дважды")
        new_tokens = {
            token
            for pattern in scenario.patterns
            for step in pattern
            for token in step_tokens(step)
        } - self._vocabulary - WILDCARDS
        vocabulary = self._vocabulary | new_tokens if new_tokens else self._vocabulary
        rules = replace(self._defaults, vocabulary=vocabulary, scenarios=(scenario,))
        if scenario.order == "sequence":
            self._insert_sequence(rules, scenario)
        else:
            known = len(self._unordered)
            insert_unordered(self._unordered, self._unordered_owners, rules)
            self._add_collectors(self._unordered[known:])
        self._cooldowns[scenario.name] = scenario.cooldown
        self._holds[scenario.name] = self._hold_of(scenario)
        self._intercepting |= self._marker_tokens & pattern_tokens((scenario,))
        self._declared[scenario.name] = len(self._declared)
        self._min_confidences[scenario.name] = rules.confidence_threshold(scenario)
        self._vocabulary = vocabulary
        self._lookback = max(self._lookback, before_of(scenario))

    def _insert_sequence(self, rules: RuleSet, scenario: Scenario) -> None:
        """Adds the paths of a `sequence` scenario to the trie and updates the indexes."""
        changes = insert_scenario(self._starts, rules, scenario)
        for old, prefix, cut in changes.splits:
            self._relocate_fronts(old, prefix, cut)
        # Концы старых рёбер сохраняют identity и silences: Window и Timer не меняются.
        for start in changes.starts:
            self._start_index.setdefault(start.token, []).append(start)
        for node, edge in changes.links:
            # новый переход из позиции с кандидатами: событие с его токеном их продолжит; срок
            # фронта в очереди теперь может быть раньше настоящего
            front = self._fronts.get(Cursor(node, len(node.segment)))
            if front is not None:
                self._link(front, edge)

    def _relocate_fronts(self, old: Node, prefix: Node, cut: int) -> None:
        """Moves fronts and misses after node `old` was split (`_relocate`)."""
        # переходы перенесённой позиции прежние, поэтому индекс ожидания не меняется
        fronts = {}
        for cursor, front in self._fronts.items():
            front.place = _relocate(cursor, old, prefix, cut)
            fronts[front.place] = front
        self._fronts = fronts
        self._misses = [
            miss._replace(place=_relocate(miss.place, old, prefix, cut)) for miss in self._misses
        ]

    @property
    def vocabulary(self) -> frozenset[str]:
        """Current vocabulary, including tokens of added scenarios."""
        return self._vocabulary

    @property
    def held(self) -> list[Held]:
        """Completions waiting for release (correction mode), in completion order."""
        return [
            Held(hold.scenario, hold.partial, hold.deadline, bool(hold.waits))
            for hold in self._held
        ]

    @property
    def active_count(self) -> int:
        """Candidates in fronts, open silence windows and held completions."""
        return self._active_count

    @property
    def starts(self) -> dict[Start, Node]:
        """Edges from the root of the trie the engine walks."""
        return self._starts

    @property
    def unordered(self) -> list[Unordered]:
        """Patterns of scenarios with `order: "any"`."""
        return self._unordered

    @property
    def position(self) -> int:
        """Position the next event will get."""
        return self._position

    @property
    def time(self) -> float | None:
        """Engine time: of the last event or `advance`; None before any."""
        return self._last_time

    @property
    def misses(self) -> list[Miss]:
        """What the last event or `advance` did not advance or let fire, and why."""
        return self._misses

    def snapshot(self) -> list[Waiting]:
        """All partial matches: in trie positions, `any` sets and silence windows."""
        # Во фронтах только позиции с токеновым продолжением, в том числе внутри сегмента.
        waiting = [
            Waiting(cursor, partial, partial.last_time + cursor.max_gap)
            for cursor, front in self._fronts.items()
            for partial in front
        ]
        for collector in self._collectors:
            gap = collector.pattern.max_gap
            for filled, front in collector.states.items():
                place = collector.place(filled)
                waiting += [Waiting(place, partial, partial.last_time + gap) for partial in front]
        waiting += [
            Waiting(timer.payload.window, timer.payload.partial, timer.deadline)
            for timer in self._hushes
        ]
        return waiting

    def next_deadline(self) -> float | None:
        """Nearest moment `advance` may emit something without an event.

        Returns:
            The nearest end of a silence window or deadline of a held completion; while a
            marker is deferred, also the nearest expiry of a match. None if nothing waits.
        """
        deadline = self._wheel.next_deadline()
        times = [hold.deadline for hold in self._held if not hold.waits]
        if self._deferred:
            # отложенный маркер решается, когда умрут совпадения: сроки снятия первых во фронтах
            times += self._front_expiries()
        if times:
            first = min(times)
            deadline = first if deadline is None else min(deadline, first)
        return deadline

    def _front_expiries(self) -> list[float]:
        """Expiry moments of the head matches of all fronts."""
        return [
            expiry(front.first().last_time, _gap(front.place), isinstance(front.place, Cursor))
            for front in self._all_fronts()
            if len(front)
        ]

    def feed(self, events: Iterable[Event]) -> list[Match]:
        """Pushes events in order and returns the matches of all of them."""
        matches: list[Match] = []
        for event in events:
            matches.extend(self.push(event))
        return matches

    def advance(self, time: float) -> list[Match]:
        """Moves time without an event.

        Closes silence windows and releases held completions with a deadline earlier than
        `time`; in correction mode also removes expired matches.

        Args:
            time: new engine time.

        Returns:
            Matches emitted by the move.

        Raises:
            ValueError: `time` is not finite or is earlier than the engine time.
        """
        if not math.isfinite(time):
            raise ValueError(f"время {time} не конечное число")
        self._move(time, f"время {time}")
        self._misses = []
        matches = self._close(time)
        if self._correcting:
            # снятые по сроку решают отложенные маркеры: момент решения — когда умерло последнее
            moment = self._expire(time, None, ())
            matches += self._settle_markers(time if moment is None else moment)
        self._tick()
        return matches

    def push(self, event: Event) -> list[Match]:
        """Processes the next event of the stream.

        Args:
            event: the event; its time must not be earlier than the engine time.

        Returns:
            Matches emitted at the event, including windows and held completions due before it.

        Raises:
            ValueError: the event is earlier than the engine time.
        """
        self._move(event.time, f"время {event.time} токена {event.token!r}")
        self._misses = []
        matches = self._close(event.time)  # окна и ждущие, истёкшие раньше события
        step = Step(self._position, event)
        self._position += 1
        wild = self._prepare(step) if self._correcting else step
        # токены шагов, к которым подходит событие: по ним ищут в индексах
        keys = fitting(event.token, self._vocabulary)
        completed: dict[str, list[Partial]] = {}  # сценарий → совпадения, завершённые этим событием
        # куда входят совпадения этого события: добавляются после обхода, чтобы событие не
        # продолжило совпадение, которое само же и продлило
        entries: list[tuple[Cursor | tuple[Collector, int], Partial]] = []
        if self._windows:
            self._break_silences(event, keys)
        self._expire(event.time, event, keys)
        self._follow_fronts(keys, step, wild, completed, entries)
        self._begin_matches(keys, step, wild, completed, entries)
        if self._collector_index:
            self._collect_sets(keys, step, wild, completed, entries)
        for place, partial in entries:
            self._store(place, partial)
        # срабатывания выдаются после обхода: из совпадений сценария выбирается самое уверенное
        if completed:
            matches += self._finish(completed, event.time)
        if self._correcting:
            matches += self._after_step(step)
        self._recent.append(step)
        while self._recent and self._recent[0].event.time < event.time - self._lookback:
            self._recent.popleft()
        self._tick()
        return matches

    def _prepare(self, step: Step) -> Step:
        """Correction mode before the event's pass: resets `_stepped`, records the phrase tail.

        Returns:
            The event's step for wildcard steps: a separate object if the token is a marker
            from patterns, so that the marker intercepts only the explicit step (`has`).
        """
        self._stepped = set()
        wild = step
        if step.event.token in self._intercepting:
            wild = Step(step.position, step.event)
        if self._phrases:
            self._said.append(step)
            first = self._said[0].position
            self._used = [used for used in self._used if used.position >= first]
        return wild

    def _break_silences(self, event: Event, keys: tuple[str, ...]) -> None:
        """Cancels open silence windows that the event breaks; expired ones are closed already."""
        broken: dict[Timer[Hush], None] = {}  # окно с двумя подходящими запретами — один раз
        for key in keys:
            for timer in self._windows.get(key, ()):
                if event.score >= timer.payload.window.edge.min_score:
                    broken[timer] = None
        for timer in broken:
            window, partial = timer.payload
            self._wheel.cancel(timer)
            self._unlist(timer)
            self._misses.append(Miss("silence", window, partial, None, timer.deadline, event.time))

    def _follow_fronts(
        self,
        keys: tuple[str, ...],
        step: Step,
        wild: Step,
        completed: dict[str, list[Partial]],
        entries: list,
    ) -> None:
        """Continues the fronts that wait for the event's token, found by index."""
        token = step.event.token
        explicit = wild is not step  # иначе все шаги события — один объект
        for key in keys:
            mine = wild if explicit and key != token else step
            for front in self._waiting.get(key, ()):
                cursor = front.place
                for edge in front.edges[key]:
                    self._follow(front, edge, cursor.follow(edge), mine, completed, entries)

    def _begin_matches(
        self,
        keys: tuple[str, ...],
        step: Step,
        wild: Step,
        completed: dict[str, list[Partial]],
        entries: list,
    ) -> None:
        """Starts new matches from the root edges with the event's token."""
        event = step.event
        explicit = wild is not step
        for key in keys:
            mine = wild if explicit and key != event.token else step
            for start in self._start_index.get(key, ()):
                child = Cursor(self._starts[start], 1)
                if event.score >= start.min_score:
                    # новое совпадение задержку не проверяет
                    self._enter(child, Partial.begin(mine), completed, entries)
                    if self._stepped is not None:
                        self._stepped.add(step.position)
                else:
                    miss = Miss("score", child, None, None, start.min_score, event.score)
                    self._misses.append(miss)

    def _collect_sets(
        self,
        keys: tuple[str, ...],
        step: Step,
        wild: Step,
        completed: dict[str, list[Partial]],
        entries: list,
    ) -> None:
        """Passes the event to the `any` patterns with its token, each once."""
        seen: set[Collector] = set()  # набор с шагами `<шаги>` и `any` стоит в индексе дважды
        for key in keys:
            for collector in self._collector_index.get(key, ()):
                if collector not in seen:
                    seen.add(collector)
                    self._collect(collector, step, wild, completed, entries)

    def _after_step(self, step: Step) -> list[Match]:
        """Correction mode after the event's pass: extends deadlines, settles and hears markers.

        Returns:
            Held completions released because deferred markers were settled.
        """
        time = step.event.time
        if self._stepped:
            # шаг родственника: совпадение с тем же первым шагом продолжает фразу
            for hold in self._held:
                if hold.partial.first_position in self._stepped:
                    hold.deadline = max(hold.deadline, time + hold.hold)
            negation = self._negation
            if negation is not None:
                negation.deadline = max(negation.deadline, time + self._defaults.max_gap)
        matches = self._settle_markers(time)
        if self._phrases:
            self._hear(step)
        return matches

    def _hear(self, step: Step) -> None:
        """Markers that end with event `step`: they act at once, are cancelled or deferred."""
        heard = self._heard()
        for kind in ("correction", "break", "negation"):
            if kind not in heard:
                continue
            mine = [item for item in heard[kind] if item.event.token in self._intercepting]
            if any(item is used for item in mine for used in self._used):
                continue  # маркер — шаг сработавшего совпадения: действие отменено
            if mine and self._alive(mine):
                self._defer(kind, mine, step)
            else:
                self._act(kind, step)

    def _act(self, kind: str, step: Step) -> None:
        """Applies a marker that ended with `step`.

        A correction cancels held completions and closes the negation, a break closes the
        negation, a negation opens or extends.
        """
        time = step.event.time
        if kind == "correction":
            self._cancel(self._held, time, "corrected")
            self._negation = None
        elif kind == "break":
            self._negation = None
        else:
            deadline = time + self._defaults.max_gap
            if self._negation is None:
                self._negation = Negation(step.position, time, deadline, None)
            else:
                self._negation.deadline = max(self._negation.deadline, deadline)

    def _defer(self, kind: str, steps: list[Step], step: Step) -> None:
        """Defers a marker whose tokens are steps of live matches.

        A correction freezes held completions and the open negation, a break freezes the
        negation, a negation opens tentatively.

        Args:
            kind: what the marker does.
            steps: steps of the phrase with tokens from patterns.
            step: the event that ended the phrase.
        """
        time = step.event.time
        marker = Marker(kind, steps, time)
        negation = self._negation
        if kind == "correction":
            if not self._held and negation is None:
                return  # снимать и закрывать нечего
            marker.holds = list(self._held)
            for hold in self._held:
                hold.waits.add(marker)
            if negation is not None:
                negation.breaks.append(marker)
        elif kind == "break":
            if negation is None:
                return
            negation.breaks.append(marker)
        else:
            deadline = time + self._defaults.max_gap
            if negation is not None:
                negation.deadline = max(negation.deadline, deadline)
                return
            self._negation = Negation(step.position, time, deadline, marker)
        self._deferred.append(marker)

    def _alive(self, steps: list[Step]) -> bool:
        """Whether one of the marker's steps is an explicit step of a live match."""
        partials = chain(
            (partial for front in self._all_fronts() for partial in front),
            (timer.payload.partial for timer in self._hushes),
        )
        return any(has(partial, steps) for partial in partials)

    def _witness(self, partials: list[Partial], time: float) -> None:
        """Matches of the moment that will fire: markers that became their steps are cancelled."""
        said = [step for step in self._said if step.event.token in self._intercepting]
        self._used += [step for step in said if any(has(partial, [step]) for partial in partials)]
        for marker in list(self._deferred):
            if any(has(partial, marker.steps) for partial in partials):
                self._settle(marker, "cancelled", time)

    def _settle_markers(self, moment: float) -> list[Match]:
        """Applies at `moment` the deferred markers whose matches all died.

        Returns:
            Held completions that nothing holds any more and whose deadline has passed.
        """
        for marker in list(self._deferred):
            if not self._alive(marker.steps):
                self._settle(marker, "applied", moment)
        now = self._last_time
        return self._release(lambda due: due < now) if self._held else []

    def _unfreeze(self, hold: Hold, wait: "Marker | Victims", moment: float) -> None:
        """`wait` no longer holds `hold`; released after its deadline, it goes after `moment`."""
        hold.waits.discard(wait)
        if not hold.waits:
            hold.deadline = max(hold.deadline, moment)

    def _settle(self, marker: Marker, state: str, moment: float) -> None:
        """Resolves a deferred marker.

        Args:
            marker: the marker.
            state: "applied" or "cancelled".
            moment: time of the decision.
        """
        marker.state = state
        self._deferred.remove(marker)
        if marker.kind == "correction":
            frozen = [hold for hold in marker.holds if hold in self._held]
            if state == "applied":
                self._cancel(frozen, moment, "corrected")
            else:
                for hold in frozen:
                    self._unfreeze(hold, marker, moment)
        self._settle_negation(marker, state)
        for group in list(self._groups):
            self._judge(group, moment)

    def _settle_negation(self, marker: Marker, state: str) -> None:
        """The open negation after `marker` was resolved: it opens for sure, closes or stays."""
        negation = self._negation
        if negation is None:
            return
        if negation.marker is marker:
            if state == "cancelled":
                self._negation = None
            else:
                negation.marker = None
        elif marker in negation.breaks:
            if state == "applied":
                self._negation = None
            else:
                negation.breaks.remove(marker)

    def _judge(self, group: Victims, moment: float) -> None:
        """Suppresses the group's completions or releases them once its markers are resolved."""
        marker, breaks = group.marker, group.breaks
        if (marker is not None and marker.state == "cancelled") or any(
            item.state == "applied" for item in breaks
        ):
            for hold in group.holds:
                self._unfreeze(hold, group, moment)
        elif (marker is None or marker.state == "applied") and all(
            item.state == "cancelled" for item in breaks
        ):
            victims = [hold for hold in group.holds if hold in self._held]
            self._cancel(victims, moment, "negated", group.time)
        else:
            return
        self._groups.remove(group)

    def _all_fronts(self) -> list[Front]:
        """Fronts of trie positions and of `any` set states."""
        return [
            *self._fronts.values(),
            *(front for collector in self._collectors for front in collector.states.values()),
        ]

    def _heard(self) -> dict[str, list[Step]]:
        """Marker phrases that end with the last event.

        Returns:
            What the phrases do → steps of their tokens.
        """
        said = list(self._said)
        heard: dict[str, list[Step]] = {}
        for length in self._lengths:
            if length > len(said):
                break
            tail = said[-length:]
            kind = self._phrases.get(tuple(step.event.token for step in tail))
            if kind is not None and all(
                step.event.score >= self._defaults.min_score for step in tail
            ):
                heard.setdefault(kind, []).extend(tail)
        return heard

    def _cancel(
        self, holds: list[Hold], time: float, reason: str, need: float | None = None
    ) -> None:
        """Removes held completions as if they never fired: tokens and cooldown are freed.

        Args:
            holds: the completions.
            time: the moment of removal.
            reason: "corrected" or "negated".
            need: `need` of the misses; None — the deadline of each completion.
        """
        if not holds:
            return
        gone = set(holds)
        self._held = [hold for hold in self._held if hold not in gone]
        for hold in holds:
            self._taken[hold.scenario].difference_update(
                step.position for step in hold.partial.backwards()
            )
            self._taken_count -= hold.partial.count
            self._active_count -= 1
            missed = hold.deadline if need is None else need
            self._misses.append(Miss(reason, None, hold.partial, hold.scenario, missed, time))
        self._kept = min(self._kept, self._taken_count)
        self._restore_cooldowns({hold.scenario for hold in holds})
        self._withdraw([hold.partial for hold in holds], time, reason)

    def _restore_cooldowns(self, scenarios: set[str]) -> None:
        """Cooldown of each scenario returns to its latest remaining completion."""
        for scenario in scenarios:
            times = [hold.time for hold in self._held if hold.scenario == scenario]
            if scenario in self._settled_at:
                times.append(self._settled_at[scenario])
            if times:
                self._fired_at[scenario] = max(times)
            else:
                del self._fired_at[scenario]

    def _withdraw(self, partials: list[Partial], time: float, reason: str) -> None:
        """Cleans up after removed completions `partials` at `time`.

        Matches and silence windows with the last token of a removed completion are removed;
        copies of its prefixes measure the gap from `time`; other matches do not change.
        """
        lasts = {partial.tail.step.position for partial in partials}
        low = min(lasts)
        prefixes = _prefixes(partials)
        # совпадения с последним токеном снятого уходят: иначе «открой шторы, ой, …, разогрей
        # чайник» собрало бы «утро»
        for front in self._all_fronts():
            self._rework(front, lasts, low, prefixes, time, reason)
        for timer in [timer for timer in self._hushes if _shares(timer.payload.partial, lasts, low)]:
            window, partial = timer.payload
            self._wheel.cancel(timer)
            self._unlist(timer)
            self._misses.append(Miss(reason, window, partial, None, time, time))

    def _rework(
        self,
        front: Front,
        lasts: set[int],
        low: int,
        prefixes: dict[int, set[tuple[int, ...]]],
        time: float,
        reason: str,
    ) -> None:
        """Removes matches with one of `lasts` from the front and restarts prefix copies.

        Matches with a position of `lasts` (`_shares`) are removed; copies of removed prefixes
        (`_copies`) measure the gap from `time`. An emptied front stays in place until its
        entry in the deadline queue comes up.
        """
        kept: list[Partial] = []
        moved: list[Partial] = []
        for partial in front:
            if _shares(partial, lasts, low):
                self._misses.append(Miss(reason, _shown(front.place), partial, None, time, time))
            elif _copies(partial, prefixes) and partial.last_time < time:
                moved.append(partial._replace(last_time=time))
            else:
                kept.append(partial)
        if len(kept) == len(front):
            return
        self._active_count -= len(front) - len(kept)
        front.items, front.head = kept, 0
        for partial in moved:
            self._active_count += front.insert(partial)

    def _move(self, time: float, what: str) -> None:
        """Sets engine time.

        Args:
            time: new time.
            what: what brings the time, for the error message.

        Raises:
            ValueError: `time` is earlier than the engine time.
        """
        if self._last_time is not None and time < self._last_time:
            raise ValueError(f"{what} раньше предыдущего {self._last_time}")
        self._last_time = time

    def _add_collectors(self, patterns: Iterable[Unordered]) -> None:
        """Adds `any` patterns and puts them into the token index."""
        for pattern in patterns:
            collector = Collector(pattern)
            self._collectors.append(collector)
            for token in collector.groups:
                self._collector_index.setdefault(token, []).append(collector)

    def _link(self, front: Front, edge: Edge) -> None:
        """Transition of the front's position: an event with its token finds the front by index."""
        front.edges.setdefault(edge.token, []).append(edge)
        self._waiting.setdefault(edge.token, {})[front] = None

    def _store(self, place: Cursor | tuple[Collector, int], partial: Partial) -> None:
        """Puts a candidate into the front of its place; a new front joins the indexes."""
        if isinstance(place, Cursor):
            front = self._fronts.get(place)
            if front is None:
                front = self._fronts[place] = Front(place)
                for edge, _ in place.transitions():
                    self._link(front, edge)
        else:
            collector, filled = place
            front = collector.states.get(filled)
            if front is None:
                front = collector.states[filled] = Front(place)
        self._active_count += front.add(partial)
        if not front.queued:
            self._schedule(front)

    def _schedule(self, front: Front) -> None:
        """Queues a non-empty front by the deadline of its oldest candidate."""
        due = due_time(front.first().last_time, _gap(front.place))
        heapq.heappush(self._deadlines, (due, next(self._order), front))
        front.queued = True

    def _expire(self, now: float, event: Event | None, keys: tuple[str, ...]) -> float | None:
        """Removes candidates whose deadline passed by `now`: the heads of due fronts.

        Args:
            now: current time.
            event: the current event; None for a call from `advance`.
            keys: step tokens the event fits: removal explains lateness on such transitions.

        Returns:
            Without an event, the latest expiry of the removed candidates, when the last of
            them died; otherwise, or if nothing was removed, None.
        """
        latest: float | None = None
        due: list[Front] = []
        while self._deadlines and self._deadlines[0][0] <= now:
            due.append(heapq.heappop(self._deadlines)[2])
        for front in due:
            front.queued = False
            if isinstance(front.place, Cursor):
                latest = self._expire_position(front, now, event, keys, latest)
            else:
                latest = self._expire_state(front, now, event, latest)
            if len(front):
                self._schedule(front)
            else:
                self._drop(front)
        return latest

    def _expire_position(
        self,
        front: Front,
        now: float,
        event: Event | None,
        keys: tuple[str, ...],
        latest: float | None,
    ) -> float | None:
        """`_expire` for a trie position; returns `latest` updated by the removed candidates."""
        place = front.place
        gap = place.max_gap
        # переходы с токеном события: снятие объясняет опоздание на них
        edges = [edge for key in keys for edge in front.edges.get(key, ())]
        while len(front) and now - front.first().last_time > gap:
            partial = front.pop_first()
            last = partial.last_time
            self._misses += [
                Miss("late", place.follow(edge), partial, None, last + edge.max_gap, now)
                for edge in edges
            ] or [Miss("expired", place, partial, None, last + gap, now)]
            self._active_count -= 1
            if event is None:
                latest = max(expiry(last, gap, True), latest or -math.inf)
        return latest

    def _expire_state(
        self, front: Front, now: float, event: Event | None, latest: float | None
    ) -> float | None:
        """`_expire` for an `any` set state; returns `latest` updated by the removed candidates."""
        collector, filled = front.place
        gap = collector.pattern.max_gap
        reason = None
        while len(front) and now > front.first().last_time + gap:
            if reason is None:
                # задержка у открытых шагов одна: «опоздал», если событие закрыло бы шаг
                if event is None:
                    reason = "expired"
                else:
                    groups = collector.fitting(event.token, self._vocabulary)
                    late = any(mask & ~filled for _, mask in groups)
                    reason = "late" if late else "expired"
            partial = front.pop_first()
            deadline = partial.last_time + gap
            self._misses.append(Miss(reason, collector.place(filled), partial, None, deadline, now))
            self._active_count -= 1
            if event is None:
                latest = max(deadline, latest or -math.inf)
        return latest

    def _drop(self, front: Front) -> None:
        """Removes an empty front from its place and from the waiting index."""
        place = front.place
        if isinstance(place, Cursor):
            del self._fronts[place]
            for token in front.edges:
                waiting = self._waiting[token]
                del waiting[front]
                if not waiting:
                    del self._waiting[token]
        else:
            del place[0].states[place[1]]

    def _follow(
        self,
        front: Front,
        edge: Edge,
        child: Cursor,
        step: Step,
        completed: dict[str, list[Partial]],
        entries: list,
    ) -> None:
        """The event along transition `edge` from the front's position.

        Expired candidates are removed already.

        Args:
            front: candidates of the position.
            edge: the transition.
            child: position behind the transition.
            step: the event's step for this transition.
            completed: scenario → matches completed by the event; filled here.
            entries: places and matches to store after the pass; filled here.
        """
        event = step.event
        items = front.items
        # срок считается от последнего шага совпадения, а не от последнего токена потока;
        # фронт идёт от старых к свежим, поэтому в задержку ребра укладывается его хвост
        fresh = front.since(lambda partial: event.time - partial.last_time <= edge.max_gap)
        for partial in islice(items, front.head, fresh):
            deadline = partial.last_time + edge.max_gap
            self._misses.append(Miss("late", child, partial, None, deadline, event.time))
        if fresh == len(items):
            return
        if event.score < edge.min_score:
            for partial in islice(items, fresh, None):
                self._misses.append(Miss("score", child, partial, None, edge.min_score, event.score))
            return
        if self._stepped is not None:
            self._stepped.update(items[index].first_position for index in range(fresh, len(items)))
        if child.scenario is not None:
            completed.setdefault(child.scenario, []).extend(
                items[index].extend(step) for index in range(fresh, len(items))
            )
        if child.ahead:  # у вершины без продолжений совпадение не остаётся
            self._move_tail(child, items, fresh, step, entries)

    def _move_tail(
        self, child: Cursor, items: list[Partial], fresh: int, step: Step, entries: list
    ) -> None:
        """Continuations of `items[fresh:]` enter `child`: pruning, windows and the best one."""
        # ранний отсев: произведения в хвосте убывают, поэтому отсеянные — в его конце
        end = len(items)
        while end > fresh and not child.ahead.reaches(items[end - 1].product * step.event.score):
            self._prune(child, items[end - 1].extend(step))
            end -= 1
        if end == fresh:
            return
        # окна не вытесняют друг друга, и `before` смотрит на шаги самого совпадения: окно
        # открывает каждое продолжение
        for silence in child.silences():
            for index in range(fresh, end):
                self._open(Window(child.node, silence), items[index].extend(step))
        if child.offset < len(child.node.segment) or child.node.children:
            entries.append((child, best(items, fresh, end, step)))

    def _enter(
        self,
        node: Cursor,
        partial: Partial,
        completed: dict[str, list[Partial]],
        entries: list | None,
    ) -> None:
        """One match enters a position; `entries` None — straight into the front (window end)."""
        if node.scenario is not None:
            completed.setdefault(node.scenario, []).append(partial)
        # ранний отсев; у вершины без продолжений `ahead` пуст, и совпадение в ней не остаётся
        if node.ahead.reaches(partial.product):
            if node.offset < len(node.node.segment) or node.node.children:
                if entries is None:
                    self._store(node, partial)
                else:
                    entries.append((node, partial))
            for edge in node.silences():
                self._open(Window(node.node, edge), partial)
        elif node.ahead:
            self._prune(node, partial)

    def _prune(self, node: Cursor, partial: Partial) -> None:
        """Reports a pruned match against the pattern closest to its threshold."""
        # ближе всего к порогу — шаблон с наибольшим запасом
        length, threshold = max(node.ahead, key=lambda item: partial.bound(item[0]) - item[1])
        self._misses.append(Miss("pruned", node, partial, None, threshold, partial.bound(length)))

    def _collect(
        self,
        collector: Collector,
        step: Step,
        wild: Step,
        completed: dict[str, list[Partial]],
        entries: list,
    ) -> None:
        """The event in an `any` pattern: closes open steps of its states and starts new sets.

        Args:
            collector: the pattern.
            step: the event's step for explicit tokens.
            wild: the event's step for `any` and `any_vac` steps (`Engine.push`).
            completed: scenario → matches completed by the event; filled here.
            entries: places and matches to store after the pass; filled here.
        """
        event = step.event
        pattern = collector.pattern
        groups = collector.fitting(event.token, self._vocabulary)
        # снятые по сроку уже ушли, а задержка у шагов набора одна: подходят все
        for filled, front in collector.states.items():
            free, low = collector.closes(filled, groups, event.score)
            for index in free:
                target, mine = filled | 1 << index, _pick(pattern, index, step, wild)
                self._gather(collector, target, front.items, front.head, mine, completed, entries)
            for index in low:
                need = pattern.slots[index].min_score
                place = collector.place(filled)
                self._misses += [
                    Miss("score", place, partial, None, need, event.score) for partial in front
                ]
        self._begin_set(collector, groups, step, wild, completed, entries)

    def _gather(
        self,
        collector: Collector,
        target: int,
        items: list[Partial],
        start: int,
        step: Step,
        completed: dict[str, list[Partial]],
        entries: list,
    ) -> None:
        """Continuations of `items[start:]` by `step` enter state `target`."""
        pattern = collector.pattern
        if self._stepped is not None:
            self._stepped.update(items[index].first_position for index in range(start, len(items)))
        if target == collector.full:
            completed.setdefault(pattern.scenario, []).extend(
                items[index].extend(step) for index in range(start, len(items))
            )
            return
        length = len(pattern.slots)
        threshold = pattern.min_confidence
        score = step.event.score
        # ранний отсев, как в вершине бора: отсеянные — в конце хвоста
        end = len(items)
        while end > start and (items[end - 1].product * score) ** (1 / length) < threshold:
            partial = items[end - 1].extend(step)
            place = collector.place(target)
            self._misses.append(
                Miss("pruned", place, partial, None, threshold, partial.bound(length))
            )
            end -= 1
        if end > start:
            entries.append(((collector, target), best(items, start, end, step)))

    def _begin_set(
        self,
        collector: Collector,
        groups: list[tuple[float, int]],
        step: Step,
        wild: Step,
        completed: dict[str, list[Partial]],
        entries: list,
    ) -> None:
        """Starts new matches of an `any` pattern: any of its steps may come first."""
        event = step.event
        pattern = collector.pattern
        length = len(pattern.slots)
        threshold = pattern.min_confidence
        free, low = collector.closes(0, groups, event.score)
        if free and self._stepped is not None:
            self._stepped.add(step.position)
        for index in free:
            partial = Partial.begin(_pick(pattern, index, step, wild))
            target = 1 << index
            if target == collector.full:
                completed.setdefault(pattern.scenario, []).append(partial)
            elif partial.bound(length) >= threshold:
                entries.append(((collector, target), partial))
            else:
                self._misses.append(
                    Miss("pruned", collector.place(target), partial, None, threshold,
                         partial.bound(length))
                )
        for index in low:
            need = pattern.slots[index].min_score
            self._misses.append(Miss("score", collector.place(0), None, None, need, event.score))

    def _close(self, now: float) -> list[Match]:
        """Silence windows with a deadline earlier than `now`: the match goes past the edge.

        Held completions are released in deadline order mixed with windows: a completion
        that ended a window may expire before the next window.

        Returns:
            Matches of the windows and released completions.
        """
        matches: list[Match] = []
        # окна с одним сроком закрываются в один момент, как совпадения одного события
        fired = self._wheel.advance(now)
        for deadline, timers in groupby(fired, key=lambda timer: timer.deadline):
            if self._correcting:
                matches += self._release(lambda due: due <= deadline)
                self._expire_negation(deadline)
            completed: dict[str, list[Partial]] = {}
            for timer in timers:
                self._unlist(timer)
                window, partial = timer.payload
                # после тишины не бывает тишины (rules.py), поэтому окна здесь не открываются
                self._enter(window.child, partial._replace(last_time=deadline), completed, None)
            matches += self._finish(completed, deadline)
        if self._correcting:
            matches += self._release(lambda due: due < now)
            self._expire_negation(now)
        return matches

    def _expire_negation(self, now: float) -> None:
        """Closes the open negation whose deadline is earlier than `now`."""
        if self._negation is not None and now > self._negation.deadline:
            self._negation = None

    def _release(self, due: Callable[[float], bool]) -> list[Match]:
        """Releases held completions whose deadline passed (`due`), except frozen ones.

        Returns:
            Matches by deadline; one deadline is one moment, ordered inside as in `_finish`.
        """
        if not self._held:
            return []
        ready = [hold for hold in self._held if not hold.waits and due(hold.deadline)]
        if not ready:
            return []
        gone = set(ready)
        self._held = [hold for hold in self._held if hold not in gone]
        self._active_count -= len(ready)
        for hold in ready:
            self._settled_at[hold.scenario] = max(
                self._settled_at.get(hold.scenario, hold.time), hold.time
            )
        ready.sort(
            key=lambda hold: (
                hold.deadline,
                hold.partial.first_position,
                self._declared[hold.scenario],
            )
        )
        return [
            Match(hold.scenario, hold.partial.steps, hold.partial.confidence, hold.deadline)
            for hold in ready
        ]

    def _open(self, window: Window, partial: Partial) -> None:
        """Opens a silence window unless a forbidden token came within `before` before the step."""
        edge = window.edge
        deadline = partial.last_time + edge.duration
        if edge.before > 0:
            own = {step.position for step in partial.backwards()}
            # в `_recent` только события раньше текущего: само событие шага тишину не нарушает,
            # прежние шаги совпадения — тоже
            for step in self._recent:
                if (
                    step.event.time >= partial.last_time - edge.before
                    and step.position not in own
                    and self._breaks(edge, step.event)
                ):
                    miss = Miss("silence", window, partial, None, deadline, step.event.time)
                    self._misses.append(miss)
                    return
        timer = self._wheel.schedule(deadline, Hush(window, partial))
        self._hushes[timer] = None
        for token in edge.tokens:
            self._windows.setdefault(token, {})[timer] = None
        self._active_count += 1

    def _unlist(self, timer: Timer[Hush]) -> None:
        """A window closed or broke: removes it from open windows and from the index."""
        del self._hushes[timer]
        for token in timer.payload.window.edge.tokens:
            windows = self._windows[token]
            del windows[timer]
            if not windows:
                del self._windows[token]
        self._active_count -= 1

    def _breaks(self, edge: Silence, event: Event) -> bool:
        """Whether the event breaks the silence: a forbidden token at or above the threshold."""
        return event.score >= edge.min_score and any(
            fits(token, event.token, self._vocabulary) for token in edge.tokens
        )

    def _finish(self, completed: dict[str, list[Partial]], time: float) -> list[Match]:
        """Completions of one moment.

        Args:
            completed: scenario → matches completed at the moment.
            time: the moment.

        Returns:
            Matches emitted now: by first step, on an equal start by declaration order.
        """
        chosen = []  # (сценарий, его совпадение): прошли порог, повтор и cooldown
        for scenario, partials in completed.items():
            winner = self._choose(scenario, partials, time)
            if winner is not None:
                chosen.append((scenario, winner))
        if self._intercepting and chosen:
            self._witness([winner for _, winner in chosen], time)
        matches = []
        suppressed: list[Partial] = []  # подавленные отрицанием: все срабатывания момента
        victims: list[Hold] = []  # подавит отрицание, если отложенные маркеры решатся так
        for scenario, winner in chosen:
            match = self._fire(scenario, winner, time, suppressed, victims)
            if match is not None:
                matches.append(match)
        if suppressed or victims:
            self._spend_negation(suppressed, victims, time)
        matches.sort(key=lambda match: (match.steps[0].position, self._declared[match.scenario]))
        return matches

    def _spend_negation(self, suppressed: list[Partial], victims: list[Hold], time: float) -> None:
        """The negation acted at this moment and closes.

        Suppressed completions are withdrawn now; tentative victims wait for the markers.
        """
        negation = self._negation
        self._negation = None
        if victims:
            group = Victims(negation, victims)
            for hold in victims:
                hold.waits.add(group)
            self._groups.append(group)
        else:
            self._withdraw(suppressed, time, "negated")

    def _choose(self, scenario: str, partials: list[Partial], time: float) -> Partial | None:
        """Match the scenario fires with: the most confident one past the threshold and repeat.

        Returns:
            The match; None if none passed or the scenario is in cooldown.
        """
        taken = self._taken.setdefault(scenario, set())
        eligible = [
            partial
            for partial in partials
            if partial.confidence >= self._min_confidences[scenario]
            and taken.isdisjoint(step.position for step in partial.backwards())
        ]
        if not eligible:
            self._reject(scenario, partials)
            return None
        winner = _most_confident(eligible)
        fired_at = self._fired_at.get(scenario)
        if fired_at is not None and time - fired_at < self._cooldowns[scenario]:
            until = fired_at + self._cooldowns[scenario]
            self._misses.append(Miss("cooldown", None, winner, scenario, until, time))
            return None
        return winner

    def _reject(self, scenario: str, partials: list[Partial]) -> None:
        """Misses of a scenario none of whose matches passed: low confidence or a repeat."""
        # уверенность ниже порога или повтор: токен уже вошёл в выданное срабатывание
        threshold = self._min_confidences[scenario]
        for partial in partials:
            if partial.confidence < threshold:
                confidence = partial.confidence
                self._misses.append(
                    Miss("confidence", None, partial, scenario, threshold, confidence)
                )
            else:
                self._misses.append(Miss("repeat", None, partial, scenario, 0.0, 0.0))

    def _fire(
        self,
        scenario: str,
        winner: Partial,
        time: float,
        suppressed: list[Partial],
        victims: list[Hold],
    ) -> Match | None:
        """Fires a scenario: emits the match at once, holds it or lets the negation suppress it.

        Args:
            scenario: the scenario.
            winner: its chosen match.
            time: the moment.
            suppressed: matches suppressed by the negation; filled here.
            victims: held completions a tentative negation would suppress; filled here.

        Returns:
            The match to emit now; None if it is held or suppressed.
        """
        hold = self._holds[scenario]
        negation = self._negation
        tentative = False  # отрицание с отложенным маркером или разрывом
        if hold > 0 and negation is not None and winner.tail.step.position > negation.position:
            if negation.marker is None and not negation.breaks:
                self._misses.append(Miss("negated", None, winner, scenario, negation.time, time))
                suppressed.append(winner)
                return None
            tentative = True
        self._taken[scenario].update(step.position for step in winner.backwards())
        self._taken_count += winner.count  # позиции новые: проверено выше
        self._fired_at[scenario] = time
        if hold > 0:
            held = Hold(scenario, winner, time, hold)
            self._held.append(held)
            self._active_count += 1
            if tentative:
                victims.append(held)
            return None
        if self._correcting:
            self._settled_at[scenario] = time
        return Match(scenario, winner.steps, winner.confidence)

    def _hold_of(self, scenario: Scenario) -> float:
        """Release delay of the scenario; 0 outside the correction mode."""
        return self._defaults.hold_of(scenario) if self._correcting else 0.0

    def _tick(self) -> None:
        """Cleans `_taken` from time to time: O(1) per call on average.

        Cleaning starts once the calls since the last one reach the number of candidates and
        positions kept after it.
        """
        self._idle += 1
        if self._idle >= self._active_count + self._kept:
            self._forget()

    def _forget(self) -> None:
        """Forgets taken positions earlier than the start of the oldest live match.

        New completions consist of positions not earlier than that start.
        """
        # один `min` по всем местам: `default` второго `min` не учёл бы совпадения в вершинах и
        # наборах, когда открыто хоть одно окно
        fronts = self._all_fronts()
        oldest = min(
            (
                *(partial.first_position for front in fronts for partial in front),
                *(timer.payload.partial.first_position for timer in self._hushes),
                # позиции ждущего освобождаются, если поправка его снимет
                *(hold.partial.first_position for hold in self._held),
            ),
            default=self._position,
        )
        # пустые множества уходят: иначе чистка обходила бы всех когда-либо сработавших
        self._taken = {
            scenario: kept
            for scenario, taken in self._taken.items()
            if (kept := {position for position in taken if position >= oldest})
        }
        self._taken_count = self._kept = sum(map(len, self._taken.values()))
        self._idle = 0
