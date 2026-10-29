"""Проход потока токенов по графу сценариев, по шаблонам `any` и по окнам тишины.

Событие трогает только то, что его ждёт: позиции бора и наборы `any` с подходящим переходом
находятся по индексу токенов, нарушенные окна тишины — по индексу запретов, снятые по сроку
кандидаты — по очереди сроков. Кандидатов одного места хранит `Front`: в нём лучшее
продолжение и истекающие сроки лежат с краёв.
"""

import heapq
import math
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import count, groupby, islice
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
    token: str
    score: float  # уверенность детектора: шаг принимает токен, только если она не ниже порога
    time: float  # секунды: когда токен сказан, по одной шкале для всего потока (ставит детектор)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"уверенность {self.score} токена {self.token!r} вне [0, 1]")
        if not math.isfinite(self.time):
            raise ValueError(f"время {self.time} токена {self.token!r} не конечное число")


class Step(NamedTuple):
    position: int  # сквозной номер события в потоке, между порциями не сбрасывается
    event: Event


@dataclass(frozen=True)
class Match:
    scenario: str
    steps: tuple[Step, ...]  # шаги с токенами; шаги-тишины в срабатывание не входят
    confidence: float  # общая уверенность: среднее геометрическое score шагов


@dataclass(frozen=True, slots=True)
class History:
    """Неизменяемое звено: кандидаты разделяют общие префиксы истории."""

    step: Step
    previous: "History | None" = None


class Partial(NamedTuple):
    """История событий отдельно от времени продолжения после токена или тишины."""

    tail: History
    product: float
    count: int
    first_position: int
    last_time: float

    @classmethod
    def begin(cls, step: Step) -> "Partial":
        return cls(History(step), step.event.score, 1, step.position, step.event.time)

    def backwards(self) -> Iterator[Step]:
        link = self.tail
        while link is not None:
            yield link.step
            link = link.previous

    @property
    def steps(self) -> tuple[Step, ...]:
        """Полная история материализуется только для результата или интерфейса."""
        steps = list(self.backwards())
        steps.reverse()
        return tuple(steps)

    def positions(self) -> list[int]:
        """Позиции шагов по порядку: для ничьей, когда равны уверенность и начало."""
        positions = [step.position for step in self.backwards()]
        positions.reverse()
        return positions

    @property
    def confidence(self) -> float:
        """Среднее геометрическое score шагов."""
        return self.product ** (1 / self.count)

    def bound(self, length: int) -> float:
        """Уверенность в шаблоне длины `length`, если у оставшихся шагов score 1.0: выше не бывает."""
        return self.product ** (1 / length)

    def extend(self, step: Step) -> "Partial":
        return Partial(
            History(step, self.tail),
            self.product * step.event.score,
            self.count + 1,
            self.first_position,
            step.event.time,
        )


class Window(NamedTuple):
    """Окно тишины: вершина и ребро-тишина из неё."""

    node: Node
    edge: Silence

    @property
    def child(self) -> Cursor:
        """Куда совпадение пойдёт, если окно пройдёт без запрещённых токенов."""
        return Cursor(self.node.silences[self.edge], 1)


class Hush(NamedTuple):
    """Нагрузка таймера: совпадение ждёт, что окно пройдёт без запрещённых токенов."""

    window: Window
    partial: Partial


# Где стоит совпадение: курсор radix trie, набор `any` или окно тишины.
Place = Cursor | tuple[Unordered, frozenset[int]] | Window


class Waiting(NamedTuple):
    """Частичное совпадение в снимке движка: где оно стоит и до какого момента ждёт."""

    place: Place
    partial: Partial
    until: float  # секунды: срок продолжения, а у окна тишины — её конец


class Miss(NamedTuple):
    """Почему событие не продвинуло совпадение или не дало сработать сценарию.

    - `"score"`: score токена `got` ниже порога шага `need`. `place` — куда не пустило: вершина
      за ребром или состояние набора `any`.
    - `"late"`: токен подошёл бы, но пришёл в `got`, а ждали до `need`. `place` — как у `"score"`.
    - `"expired"`: совпадение снято: ждали до `need`, а токен в `got` ему не подходил. `place` —
      где оно стояло.
    - `"pruned"`: совпадение отсеяно: даже при score 1.0 у оставшихся шагов уверенность не выше
      `got`, а порог `need`. `place` — куда оно вошло.
    - `"silence"`: тишина нарушена: запрещённый токен пришёл в `got`, а окно шло до `need`. Если
      `got` раньше последнего шага совпадения, токен пришёл до шага. `place` — окно тишины.
    - `"confidence"`: шаблон сценария собран, но общая уверенность `got` ниже порога `need`.
    - `"repeat"`: шаблон собран, но его токен уже вошёл в выданное срабатывание сценария.
    - `"cooldown"`: шаблон собран, но сценарий молчит до `need`.

    `partial` — совпадение, которое не продвинулось; `None` — новое совпадение не началось.
    У трёх последних причин есть `scenario`, а `place` — `None`.
    """

    reason: str
    place: Place | None
    partial: Partial | None
    scenario: str | None
    need: float
    got: float


def later(a: Partial, b: Partial) -> bool:
    """Шаги `a` не раньше шагов `b`: позиции сравниваются целиком, слева направо.

    Начало сравнивается сразу, а историю собирает только ничья по началу.
    """
    if a.first_position != b.first_position:
        return a.first_position > b.first_position
    return a.positions() >= b.positions()


def dominates(a: Partial, b: Partial) -> bool:
    """`a` вытесняет `b` из вершины: последний шаг у `a` не раньше, а произведение не меньше.

    Тогда любое продолжение `b` есть и у `a`, со сроком не раньше и уверенностью не ниже. Если
    равны и последний шаг, и произведение, остаётся то, у кого шаги позже (`later`): так исход
    не зависит от того, в каком порядке совпадения пришли в вершину.
    """
    if (a.tail.step.position, a.product) == (b.tail.step.position, b.product):
        return later(a, b)
    return a.tail.step.position >= b.tail.step.position and a.product >= b.product


EMPTY: dict = {}  # переходы фронта набора `any`: их нет, словарь не заводится


class Front:
    """Несравнимые совпадения одного места: позиции бора или состояния набора `any`.

    Совпадения идут по возрастанию последнего шага, а произведение score у них строго убывает:
    более старое остаётся, только если оно увереннее (`dominates`). Время продолжения растёт
    вместе с последним шагом: после тишины это срок окна, а у окон одного ребра длина одна.
    Поэтому новое совпадение вытесняет только хвост, сроки истекают с головы, в задержку ребра
    укладывается хвост, а самое уверенное в нём стоит первым.
    """

    __slots__ = ("place", "items", "head", "edges")

    def __init__(self, place: "Cursor | tuple[Collector, int]") -> None:
        self.place = place
        self.items: list[Partial | None] = []  # до `head` — None: снятые не держат историю
        self.head = 0  # совпадения до `head` сняты по сроку
        # у позиции бора: токен перехода → переходы с ним; по этим токенам фронт стоит в индексе
        self.edges: dict[str, list[Edge]] = {} if isinstance(place, Cursor) else EMPTY

    def __len__(self) -> int:
        return len(self.items) - self.head

    def __iter__(self) -> Iterator[Partial]:
        return islice(self.items, self.head, None)

    def first(self) -> Partial:
        return self.items[self.head]

    def add(self, partial: Partial) -> int:
        """Добавляет совпадение с последним шагом не раньше прочих; возвращает рост фронта.

        Вытеснить новое может только последнее: лишь у него последний шаг может быть тем же.
        """
        items, head = self.items, self.head
        if len(items) > head and dominates(items[-1], partial):
            return 0
        size = len(items)
        while len(items) > head and dominates(partial, items[-1]):
            items.pop()
        items.append(partial)
        return len(items) - size

    def pop_first(self) -> Partial:
        partial = self.items[self.head]
        self.items[self.head] = None
        self.head += 1
        if self.head * 2 >= len(self.items):  # сжатие: в среднем O(1) на снятое совпадение
            del self.items[: self.head]
            self.head = 0
        return partial

    def since(self, fresh: Callable[[Partial], bool]) -> int:
        """Номер первого совпадения, для которого `fresh` истинно; дальше оно истинно у всех."""
        return bisect_left(self.items, True, self.head, key=fresh)


class Collector:
    """Шаблон `any` в движке: закрытые шаги — битовая маска, одинаковые шаги — одна группа."""

    __slots__ = ("pattern", "groups", "full", "states")

    def __init__(self, pattern: Unordered) -> None:
        self.pattern = pattern
        self.groups = pattern.groups()  # токен шага → (порог, маска номеров) групп
        self.full = (1 << len(pattern.slots)) - 1  # маска собранного шаблона
        self.states: dict[int, Front] = {}  # маска закрытых шагов → совпадения, без пустых

    def fitting(self, token: str, vocabulary: frozenset[str]) -> list[tuple[float, int]]:
        """Группы шагов, к которым подходит токен события: одни на все состояния набора."""
        return [group for key in fitting(token, vocabulary) for group in self.groups.get(key, ())]

    def closes(
        self, filled: int, groups: list[tuple[float, int]], score: float
    ) -> tuple[list[int], list[int]]:
        """`Unordered.free` на масках: шаги, которые событие закроет, и открытые шаги с его
        токеном, порог которых оно не прошло."""
        free, low = [], []
        for min_score, mask in groups:
            open_ = mask & ~filled
            if open_:
                index = (open_ & -open_).bit_length() - 1
                (free if score >= min_score else low).append(index)
        return free, low

    def place(self, filled: int) -> tuple[Unordered, frozenset[int]]:
        """Состояние так, как его видят снимок и отказы: шаблон и номера закрытых шагов."""
        slots = range(len(self.pattern.slots))
        return self.pattern, frozenset(index for index in slots if filled >> index & 1)


def before_of(scenario: Scenario) -> float:
    """Самое длинное окно тишины назад у сценария; 0, если назад не смотрит ни одна тишина."""
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
    """Срок в очереди: на несколько ulp раньше `last + gap`.

    Снятие решает `now - last > gap`, а у вычитания своё округление: совпадение, которое оно
    снимет, должно выйти из очереди не позже. Вышедшее раньше вернётся в очередь.
    """
    return last + gap - 4 * (math.ulp(last) + math.ulp(gap))


def best(items: list[Partial], start: int, end: int, step: Step) -> Partial:
    """Лучшее продолжение `items[start:end]` событием: в месте после события останется оно одно.

    У продолжений один последний шаг, поэтому вытеснение решает произведение, а при равном —
    `later`. Произведения в хвосте фронта убывают, так что равные лучшему стоят сразу за ним:
    после score 0 равны все.
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


class Engine:
    """Частичные совпадения хранятся между вызовами, поэтому поток можно подавать порциями.

    Токеновые цепочки без развилок, завершений и тишин хранятся сжатыми. Кандидат занимает
    Cursor внутри сегмента; история разделяется по неизменяемым ссылкам History. Время
    продолжения last_time независимо от истории: после тишины это срок окна, а не новый Step.

    - Токен, не продолжающий частичное совпадение, пропускается: совпадение ждёт дальше.
      Это относится и к токенам вне словаря: их позиция и время учитываются как обычно.
    - У ребра есть задержка: совпадение продолжается по нему, только если от последнего шага
      совпадения прошло не больше неё. Посторонние токены этот срок не продлевают. Когда сроки
      всех рёбер из вершины истекли, совпадение снимается. Новое совпадение от корня задержку
      не проверяет.
    - У ребра есть порог: токен проходит по нему, только если его score не ниже порога. Иначе
      для этого ребра токена нет, и совпадение ждёт дальше. Новое совпадение тоже начинается,
      только если score токена не ниже порога ребра из корня.
    - Совпадение, продолжившееся по ребру, остаётся и в старой вершине: копия ждёт другого
      продолжения. Поэтому лишний токен со своим ребром не уводит совпадение в чужую ветку.
    - Уверенность совпадения — среднее геометрическое score его шагов. Сценарий срабатывает,
      только если она не ниже `min_confidence` сценария.
    - В вершине стоят только несравнимые совпадения: одно вытесняет другое, если у него
      последний шаг не раньше, а произведение score не меньше (`dominates`). Поэтому рядом могут
      стоять свежее совпадение и более уверенное.
    - Совпадение не остаётся в вершине, если даже при score 1.0 у всех оставшихся шагов оно не
      дотянет до порога ни одного сценария, чей шаблон кончается ниже (ранний отсев).
    - Сценарий срабатывает при входе в вершину, где заканчивается один из его шаблонов. Если
      из неё есть продолжения (более длинные шаблоны с тем же началом), совпадение остаётся
      активным.
    - Шаблоны сценариев с `order: "any"` идут мимо бора. Их совпадение — набор закрытых шагов:
      токен закрывает любой ещё открытый шаг со своим токеном, если проходит его порог и
      укладывается в задержку от последнего шага. `<дверь> <шаги>` и `<шаги> <дверь>` дают одно
      состояние `{<дверь>, <шаги>}`. Копия, вытеснение и ранний отсев — как в вершине бора;
      совпадение снимается, когда его задержка истекла.
    - Токен шага `any` подходит к любому токену события, в словаре он или нет; `any_vac` — к
      любому токену текущего словаря. Это верно везде: в бору, в наборах `any` и в запретах
      тишины.
    - Ребро-тишина: совпадение, вошедшее в вершину в момент t, открывает окно до t + `for`.
      Срок окна ставится в колесо таймеров. Запрещённый токен со score не ниже порога в окне
      нарушает тишину. Если у тишины `before` > 0, её нарушает и событие до t, не раньше
      t − `before`, кроме шагов самого совпадения. Окно, прошедшее без запрещённого токена,
      закрывается, когда время станет больше t + `for`: от события или из `advance`. Тогда
      совпадение идёт за ребро, и временем его последнего шага считается t + `for`. Вытеснения
      в окнах нет: окна сдвинуты, и токен может нарушить одно, не задев другое.
    - `push` сначала закрывает окна, истёкшие раньше события, потом разбирает само событие.
      Окна с одним сроком закрываются в один момент; сработавшие на нём сценарии получают
      время конца окна (для cooldown).
    - Если на одном событии завершились несколько совпадений одного сценария, выдаётся не
      больше одного: последний токен у них общий. Выдаётся самое уверенное, при равной
      уверенности — начавшееся позже, а при равном начале — с более поздними шагами.
    - Срабатывания одного момента (события или группы окон с одним сроком) выдаются от
      начавшегося раньше, при равном начале — в порядке объявления сценариев; добавленные
      через `add_scenario` объявлены после прежних. Моменты идут по времени.
    - Срабатывание не выдаётся, если у него есть общий токен с выданным срабатыванием того же
      сценария (токены сравниваются по позиции в потоке) или если с выданного срабатывания
      этого сценария прошло меньше его `cooldown`. Невыданное срабатывание не занимает токены
      и не продлевает cooldown, а совпадение живёт дальше.
    - Время событий и `advance` не убывает: время раньше предыдущего — ошибка.
    - Событие обходит не всех кандидатов, а только ждущих его токен, поэтому порядок записей
      `snapshot` и `misses` в контракт не входит: содержимое и кратность те же, что у полного
      обхода (tests/reference.py).
    - `add_scenario` дописывает пути в общий бор. Активные совпадения могут продолжаться по
      добавленным веткам; окна тишины открываются при входе в вершину после добавления.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._defaults = rules  # стандартные задержка и пороги не меняются
        self._vocabulary = rules.vocabulary
        self._cooldowns = {scenario.name: scenario.cooldown for scenario in rules.scenarios}
        # сценарий → номер объявления: срабатывания одного момента с одним началом идут так
        self._declared = {scenario.name: index for index, scenario in enumerate(rules.scenarios)}
        self._min_confidences = {
            scenario.name: rules.confidence_threshold(scenario) for scenario in rules.scenarios
        }
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
        self._active_count = 0  # кандидаты во фронтах и открытые окна
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

    def add_scenario(self, scenario: Scenario | dict) -> None:
        """Добавляет сценарий без перестроения графа и сброса текущих совпадений.

        Принимает `Scenario` или словарь в формате элемента `scenarios` из JSON. Проверяются
        только новые шаблоны; их токены добавляются в словарь после успешной вставки.
        Сохранившиеся префиксы доступны новому сценарию. События не переигрываются;
        снятые совпадения не восстанавливаются.
        Вызывать между `push`/`feed`; одновременные вызовы из разных потоков не поддерживаются.
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
            changes = insert_scenario(self._starts, rules, scenario)
            for old, prefix, cut in changes.splits:

                def relocate(place: Place | None) -> Place | None:
                    if isinstance(place, Cursor) and place.node is old:
                        return (
                            Cursor(prefix, place.offset)
                            if place.offset <= cut
                            else Cursor(old, place.offset - cut)
                        )
                    return place

                # переходы перенесённой позиции прежние, поэтому индекс ожидания не меняется
                fronts = {}
                for cursor, front in self._fronts.items():
                    front.place = relocate(cursor)
                    fronts[front.place] = front
                self._fronts = fronts
                self._misses = [miss._replace(place=relocate(miss.place)) for miss in self._misses]
            # Концы старых рёбер сохраняют identity и silences: Window и Timer не меняются.
            for start in changes.starts:
                self._start_index.setdefault(start.token, []).append(start)
            for node, edge in changes.links:
                # новый переход из позиции с кандидатами: событие с его токеном их продолжит; срок
                # фронта в очереди теперь может быть раньше настоящего
                front = self._fronts.get(Cursor(node, len(node.segment)))
                if front is not None:
                    self._link(front, edge)
        else:
            known = len(self._unordered)
            insert_unordered(self._unordered, self._unordered_owners, rules)
            self._add_collectors(self._unordered[known:])
        self._cooldowns[scenario.name] = scenario.cooldown
        self._declared[scenario.name] = len(self._declared)
        self._min_confidences[scenario.name] = rules.confidence_threshold(scenario)
        self._vocabulary = vocabulary
        self._lookback = max(self._lookback, before_of(scenario))

    @property
    def vocabulary(self) -> frozenset[str]:
        """Текущий словарь, включая токены успешно добавленных сценариев."""
        return self._vocabulary

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def starts(self) -> dict[Start, Node]:
        """Рёбра из корня бора, по которому идёт движок."""
        return self._starts

    @property
    def unordered(self) -> list[Unordered]:
        return self._unordered

    @property
    def position(self) -> int:
        """Номер, который получит следующее событие."""
        return self._position

    @property
    def time(self) -> float | None:
        """Текущее время движка: последнего события или `advance`."""
        return self._last_time

    @property
    def misses(self) -> list[Miss]:
        """Почему последнее событие или `advance` что-то не продвинули или не дали сработать."""
        return self._misses

    def snapshot(self) -> list[Waiting]:
        """Все частичные совпадения: в вершинах бора, в наборах `any` и в окнах тишины."""
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
        """Ближайший конец окна тишины: когда `advance` выдаст что-то без события. None — окон нет."""
        return self._wheel.next_deadline()

    def feed(self, events: Iterable[Event]) -> list[Match]:
        matches: list[Match] = []
        for event in events:
            matches.extend(self.push(event))
        return matches

    def advance(self, time: float) -> list[Match]:
        """Двигает время без события: закрывает окна тишины со сроком раньше `time`."""
        if not math.isfinite(time):
            raise ValueError(f"время {time} не конечное число")
        self._move(time, f"время {time}")
        self._misses = []
        matches = self._close(time)
        self._tick()
        return matches

    def push(self, event: Event) -> list[Match]:
        self._move(event.time, f"время {event.time} токена {event.token!r}")
        misses = self._misses = []
        matches = self._close(event.time)  # окна, истёкшие раньше события
        step = Step(self._position, event)
        self._position += 1
        # токены шагов, к которым подходит событие: по ним ищут в индексах
        keys = fitting(event.token, self._vocabulary)
        completed: dict[str, list[Partial]] = {}  # сценарий → совпадения, завершённые этим событием
        # куда входят совпадения этого события: добавляются после обхода, чтобы событие не
        # продолжило совпадение, которое само же и продлило
        entries: list[tuple[Cursor | tuple[Collector, int], Partial]] = []

        # нарушенная тишина: истёкшие окна закрыты выше, поэтому оставшиеся ещё идут
        broken: dict[Timer[Hush], None] = {}  # окно с двумя подходящими запретами — один раз
        for key in keys:
            for timer in self._windows.get(key, ()):
                if event.score >= timer.payload.window.edge.min_score:
                    broken[timer] = None
        for timer in broken:
            window, partial = timer.payload
            self._wheel.cancel(timer)
            self._unlist(timer)
            misses.append(Miss("silence", window, partial, None, timer.deadline, event.time))
        self._expire(event, keys)
        for key in keys:
            for front in self._waiting.get(key, ()):
                cursor = front.place
                for edge in front.edges[key]:
                    self._follow(front, edge, cursor.follow(edge), step, completed, entries)
        for key in keys:
            for start in self._start_index.get(key, ()):
                child = Cursor(self._starts[start], 1)
                if event.score >= start.min_score:
                    # новое совпадение задержку не проверяет
                    self._enter(child, Partial.begin(step), completed, entries)
                else:
                    misses.append(Miss("score", child, None, None, start.min_score, event.score))
        seen: set[Collector] = set()  # набор с шагами `<шаги>` и `any` стоит в индексе дважды
        for key in keys:
            for collector in self._collector_index.get(key, ()):
                if collector not in seen:
                    seen.add(collector)
                    self._collect(collector, step, completed, entries)
        for place, partial in entries:
            self._store(place, partial)
        # срабатывания выдаются после обхода: из совпадений сценария выбирается самое уверенное
        matches += self._finish(completed, event.time)

        self._recent.append(step)
        while self._recent and self._recent[0].event.time < event.time - self._lookback:
            self._recent.popleft()
        self._tick()
        return matches

    def _move(self, time: float, what: str) -> None:
        if self._last_time is not None and time < self._last_time:
            raise ValueError(f"{what} раньше предыдущего {self._last_time}")
        self._last_time = time

    def _add_collectors(self, patterns: Iterable[Unordered]) -> None:
        for pattern in patterns:
            collector = Collector(pattern)
            self._collectors.append(collector)
            for token in collector.groups:
                self._collector_index.setdefault(token, []).append(collector)

    def _link(self, front: Front, edge: Edge) -> None:
        """Переход позиции фронта: событие с его токеном найдёт фронт по индексу."""
        front.edges.setdefault(edge.token, []).append(edge)
        self._waiting.setdefault(edge.token, {})[front] = None

    def _store(self, place: Cursor | tuple[Collector, int], partial: Partial) -> None:
        """Кладёт кандидата во фронт места; новый фронт встаёт в индекс ожидания и в очередь."""
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
        queued = len(front) > 0  # у непустого фронта уже есть срок в очереди
        self._active_count += front.add(partial)
        if not queued:
            self._schedule(front)

    def _schedule(self, front: Front) -> None:
        """Ставит непустой фронт в очередь по сроку его самого старого кандидата."""
        place = front.place
        gap = place.max_gap if isinstance(place, Cursor) else place[0].pattern.max_gap
        due = due_time(front.first().last_time, gap)
        heapq.heappush(self._deadlines, (due, next(self._order), front))

    def _expire(self, event: Event, keys: tuple[str, ...]) -> None:
        """Снимает кандидатов, срок которых к событию истёк: во фронте это голова."""
        now = event.time
        due: list[Front] = []
        while self._deadlines and self._deadlines[0][0] <= now:
            due.append(heapq.heappop(self._deadlines)[2])
        for front in due:
            place = front.place
            if isinstance(place, Cursor):
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
            else:
                collector, filled = place
                gap = collector.pattern.max_gap
                reason = None
                while len(front) and now > front.first().last_time + gap:
                    if reason is None:
                        # задержка у открытых шагов одна: «опоздал», если событие закрыло бы шаг
                        groups = collector.fitting(event.token, self._vocabulary)
                        late = any(mask & ~filled for _, mask in groups)
                        reason = "late" if late else "expired"
                    partial = front.pop_first()
                    deadline = partial.last_time + gap
                    self._misses.append(
                        Miss(reason, collector.place(filled), partial, None, deadline, now)
                    )
                    self._active_count -= 1
            if len(front):
                self._schedule(front)
            elif isinstance(place, Cursor):
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
        """Событие по переходу `edge` из позиции фронта; снятые по сроку уже ушли."""
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
        if child.scenario is not None:
            completed.setdefault(child.scenario, []).extend(
                items[index].extend(step) for index in range(fresh, len(items))
            )
        targets = child.ahead
        if not targets:
            return  # у вершины без продолжений совпадение не остаётся
        # ранний отсев: произведения в хвосте убывают, поэтому отсеянные — в его конце
        end = len(items)
        while end > fresh and not targets.reaches(items[end - 1].product * event.score):
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
        """Одно совпадение входит в позицию; `entries` None — сразу во фронт (закрытие окна)."""
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
        # ближе всего к порогу — шаблон с наибольшим запасом
        length, threshold = max(node.ahead, key=lambda item: partial.bound(item[0]) - item[1])
        self._misses.append(Miss("pruned", node, partial, None, threshold, partial.bound(length)))

    def _collect(
        self,
        collector: Collector,
        step: Step,
        completed: dict[str, list[Partial]],
        entries: list,
    ) -> None:
        """Событие в шаблоне `any`: закрывает открытые шаги у его состояний и начинает новые."""
        event = step.event
        pattern = collector.pattern
        length = len(pattern.slots)
        threshold = pattern.min_confidence

        def gather(target: int, items: list[Partial], start: int) -> None:
            """Продолжения `items[start:]` событием входят в состояние `target`."""
            if target == collector.full:
                completed.setdefault(pattern.scenario, []).extend(
                    items[index].extend(step) for index in range(start, len(items))
                )
                return
            # ранний отсев, как в вершине бора: отсеянные — в конце хвоста
            end = len(items)
            while end > start and (items[end - 1].product * event.score) ** (1 / length) < threshold:
                partial = items[end - 1].extend(step)
                place = collector.place(target)
                self._misses.append(
                    Miss("pruned", place, partial, None, threshold, partial.bound(length))
                )
                end -= 1
            if end > start:
                entries.append(((collector, target), best(items, start, end, step)))

        groups = collector.fitting(event.token, self._vocabulary)
        # снятые по сроку уже ушли, а задержка у шагов набора одна: подходят все
        for filled, front in collector.states.items():
            free, low = collector.closes(filled, groups, event.score)
            for index in free:
                gather(filled | 1 << index, front.items, front.head)
            for index in low:
                need = pattern.slots[index].min_score
                place = collector.place(filled)
                self._misses += [
                    Miss("score", place, partial, None, need, event.score) for partial in front
                ]
        # начать набор может любой его шаг
        free, low = collector.closes(0, groups, event.score)
        for index in free:
            partial = Partial.begin(step)
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
        """Окна тишины со сроком раньше `now`: тишина выдержана, совпадение идёт за ребро."""
        matches: list[Match] = []
        # окна с одним сроком закрываются в один момент, как совпадения одного события
        fired = self._wheel.advance(now)
        for deadline, timers in groupby(fired, key=lambda timer: timer.deadline):
            completed: dict[str, list[Partial]] = {}
            for timer in timers:
                self._unlist(timer)
                window, partial = timer.payload
                # после тишины не бывает тишины (rules.py), поэтому окна здесь не открываются
                self._enter(window.child, partial._replace(last_time=deadline), completed, None)
            matches += self._finish(completed, deadline)
        return matches

    def _open(self, window: Window, partial: Partial) -> None:
        """Открывает окно тишины, если запрещённого токена не было и `before` секунд до шага."""
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
        """Окно закрылось или нарушено: убирает его из открытых и из индекса запретов."""
        del self._hushes[timer]
        for token in timer.payload.window.edge.tokens:
            windows = self._windows[token]
            del windows[timer]
            if not windows:
                del self._windows[token]
        self._active_count -= 1

    def _breaks(self, edge: Silence, event: Event) -> bool:
        """Нарушает ли событие тишину: запрещённый токен со score не ниже порога."""
        return event.score >= edge.min_score and any(
            fits(token, event.token, self._vocabulary) for token in edge.tokens
        )

    def _finish(self, completed: dict[str, list[Partial]], time: float) -> list[Match]:
        """Срабатывания одного момента: от начавшегося раньше, при равном начале — по объявлению."""
        matches = []
        for scenario, partials in completed.items():
            match = self._fire(scenario, partials, time)
            if match is not None:
                matches.append(match)
        matches.sort(key=lambda match: (match.steps[0].position, self._declared[match.scenario]))
        return matches

    def _fire(self, scenario: str, partials: list[Partial], time: float) -> Match | None:
        taken = self._taken.setdefault(scenario, set())
        eligible = [
            partial
            for partial in partials
            if partial.confidence >= self._min_confidences[scenario]
            and taken.isdisjoint(step.position for step in partial.backwards())
        ]
        if not eligible:
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
            return None
        # самое уверенное; при равной уверенности — с более поздними шагами (`later`)
        best = eligible[0]
        for partial in eligible[1:]:
            if partial.confidence > best.confidence or (
                partial.confidence == best.confidence and later(partial, best)
            ):
                best = partial
        fired_at = self._fired_at.get(scenario)
        if fired_at is not None and time - fired_at < self._cooldowns[scenario]:
            until = fired_at + self._cooldowns[scenario]
            self._misses.append(Miss("cooldown", None, best, scenario, until, time))
            return None
        taken.update(step.position for step in best.backwards())
        self._taken_count += best.count  # позиции новые: проверено выше
        self._fired_at[scenario] = time
        return Match(scenario, best.steps, best.confidence)

    def _tick(self) -> None:
        """Чистит `_taken`, когда с прошлой чистки прошло не меньше вызовов, чем кандидатов и
        позиций, оставшихся после неё.

        Чистка обходит кандидатов и все позиции. Новые позиции оплачены срабатываниями, поэтому
        на вызов приходится в среднем O(1) сверх них. Порог не растёт вместе с новыми
        позициями: иначе при нескольких срабатываниях на событие чистка не наступала бы.
        """
        self._idle += 1
        if self._idle >= self._active_count + self._kept:
            self._forget()

    def _forget(self) -> None:
        """Новые срабатывания состоят из позиций не раньше начала самого старого активного
        совпадения, более ранние можно забыть.

        Лишние позиции не мешают: их нет ни в живых совпадениях, ни в будущих событиях,
        поэтому чистить можно реже, чем на каждом событии.
        """
        # один `min` по всем местам: `default` второго `min` не учёл бы совпадения в вершинах и
        # наборах, когда открыто хоть одно окно
        fronts = [
            *self._fronts.values(),
            *(front for collector in self._collectors for front in collector.states.values()),
        ]
        oldest = min(
            (
                *(partial.first_position for front in fronts for partial in front),
                *(timer.payload.partial.first_position for timer in self._hushes),
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
