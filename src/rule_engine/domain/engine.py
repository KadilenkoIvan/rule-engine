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
    # режим поправок: срок, после которого выдано отложенное срабатывание; None — выдано сразу
    released: float | None = None


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


class Held(NamedTuple):
    """Завершённое совпадение в режиме поправок: сценарий сработал, выдача ждёт срока."""

    scenario: str
    partial: Partial
    until: float  # секунды: выдаётся, когда время станет больше
    frozen: bool  # ждёт решения отложенного маркера: по сроку не выдаётся


class Marker:
    """Маркер, токен которого стоит шагом живого совпадения: действие ждёт, чем оно кончится.

    Сработает совпадение с этим шагом — маркер отменяется; умрут все такие совпадения —
    применяется так, как применился бы в свой момент.
    """

    __slots__ = ("kind", "steps", "time", "state", "holds")

    def __init__(self, kind: str, steps: list[Step], time: float) -> None:
        self.kind = kind  # "correction", "negation" или "break"
        self.steps = steps  # шаги фразы с токенами из шаблонов: те самые объекты, см. `has`
        self.time = time
        self.state: str | None = None  # None — ждёт; "applied" или "cancelled"
        self.holds: list[Hold] = []  # у поправки: ждущие на её момент


class Negation:
    """Открытое отрицание: подавит первое завершённое совпадение с шагом после него."""

    __slots__ = ("position", "time", "deadline", "marker", "breaks")

    def __init__(self, position: int, time: float, deadline: float, marker: Marker | None) -> None:
        self.position = position  # последний токен фразы отрицания
        self.time = time
        self.deadline = deadline  # закрывается, когда время станет больше
        self.marker = marker  # отложенная фраза отрицания; None — отрицание открыто точно
        self.breaks: list[Marker] = []  # отложенные разрывы и поправки: применятся — закроют


class Hold:
    """Ждущее срабатывание: срок продлевают новые шаги совпадений с тем же первым шагом."""

    __slots__ = ("scenario", "partial", "time", "hold", "deadline", "waits")

    def __init__(self, scenario: str, partial: Partial, time: float, hold: float) -> None:
        self.scenario = scenario
        self.partial = partial
        self.time = time  # момент завершения: от него считался cooldown
        self.hold = hold
        self.deadline = time + hold
        self.waits: set[Marker | Victims] = set()  # чьего решения ждёт выдача


class Victims:
    """Срабатывания одного момента, которые подавило бы отрицание с отложенным маркером или
    разрывом: подавляются, если маркер применится, а разрывы отменятся."""

    __slots__ = ("marker", "breaks", "time", "holds")

    def __init__(self, negation: Negation, holds: list[Hold]) -> None:
        self.marker = negation.marker
        self.breaks = list(negation.breaks)
        self.time = negation.time
        self.holds = holds


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
    - `"corrected"` (режим поправок): поправка в `got` сняла ждущее срабатывание `scenario`,
      которое ждало выдачи до `need`, или частичное совпадение в `place`, в которое входит
      последний токен снятого срабатывания (тогда `need` = `got`).
    - `"negated"` (режим поправок): отрицание, открытое в `need`, подавило срабатывание
      `scenario` в `got`, или снято частичное совпадение в `place`, в которое входит последний
      токен подавленного (тогда `need` = `got`).

    `partial` — совпадение, которое не продвинулось; `None` — новое совпадение не началось.
    У отказов сценария есть `scenario`, а `place` — `None`.
    """

    reason: str
    place: Place | None
    partial: Partial | None
    scenario: str | None
    need: float
    got: float


def has(partial: Partial, steps: list[Step]) -> bool:
    """Входит ли в совпадение один из шагов `steps`: тот же объект, а не равный ему.

    Шаг по явному токену и шаг по `any` — разные объекты Step одного события (`Engine.push`).
    """
    low = min(step.position for step in steps)
    for step in takewhile(lambda item: item.position >= low, partial.backwards()):
        if any(step is mine for mine in steps):
            return True
    return False


def pattern_tokens(scenarios: Iterable[Scenario]) -> set[str]:
    """Токены шагов шаблонов, без запретов тишины."""
    return {
        step.token
        for scenario in scenarios
        for pattern in scenario.patterns
        for step in pattern
        if isinstance(step, PatternStep)
    }


def later(a: Partial, b: Partial) -> bool:
    """Шаги `a` не раньше шагов `b`: позиции сравниваются целиком, слева направо.

    Начало сравнивается сразу, а историю собирает только ничья по началу.
    """
    if a.first_position != b.first_position:
        return a.first_position > b.first_position
    return a.positions() >= b.positions()


def dominates(a: Partial, b: Partial) -> bool:
    """`a` вытесняет `b` из вершины: время продолжения у `a` не раньше, при равном — последний
    шаг не раньше, а произведение не меньше.

    Тогда любое продолжение `b` есть и у `a`, со сроком не раньше и уверенностью не ниже. Если
    равны и время, и последний шаг, и произведение, остаётся то, у кого шаги позже (`later`):
    так исход не зависит от того, в каком порядке совпадения пришли в вершину. Время
    продолжения растёт вместе с последним шагом; обгоняет его только копия, которой поправка
    перенесла отсчёт.
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
    """Порядок во фронте: время продолжения, при равном — последний шаг."""
    return partial.last_time, partial.tail.step.position


EMPTY: dict = {}  # переходы фронта набора `any`: их нет, словарь не заводится


class Front:
    """Несравнимые совпадения одного места: позиции бора или состояния набора `any`.

    Совпадения идут по возрастанию времени продолжения (`mark`), а произведение score у них
    строго убывает: более старое остаётся, только если оно увереннее (`dominates`). Время
    продолжения растёт вместе с последним шагом: после тишины это срок окна, а у окон одного
    ребра длина одна. Поэтому новое совпадение вытесняет только хвост, сроки истекают с головы,
    в задержку ребра укладывается хвост, а самое уверенное в нём стоит первым.
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

    def insert(self, partial: Partial) -> int:
        """Ставит совпадение на его место по `mark`; возвращает рост фронта.

        Вытеснить новое может только первое совпадение не раньше него: у него наибольшее
        произведение среди таких. Новое вытесняет совпадения сразу перед собой и равные ему по
        `mark`.
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


def expiry(last: float, gap: float, sequence: bool) -> float:
    """Последний момент, когда совпадение ещё не снято: снимается, когда время станет больше.

    В бору снятие решает `now - last > gap`, в наборе `any` — `now > last + gap`.
    """
    deadline = last + gap
    if sequence:
        while deadline - last > gap:
            deadline = math.nextafter(deadline, -math.inf)
        while math.nextafter(deadline, math.inf) - last <= gap:
            deadline = math.nextafter(deadline, math.inf)
    return deadline


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

    Режим поправок (`self_correction` в правилах, CORRECTIONS.md). Маркер — фраза из
    `corrections`, `negations` или `negation_breaks`: её токены идут в потоке подряд, у каждого
    score не ниже `min_score` корня. Сценарии с `hold` 0 маркеры не трогают.

    - Срабатывание сценария с `hold` > 0 не выдаётся сразу, а ждёт: срок — момент завершения
      плюс `hold`. Срок продлевает новый шаг любого совпадения с тем же первым шагом: до
      времени этого шага плюс `hold`. Когда время станет больше срока, срабатывание выдают
      `push` или `advance`; ждущие с одним сроком выдаются в один момент, раньше окон тишины
      с тем же сроком. Повтор и cooldown проверяются при завершении: ждущее срабатывание
      занимает токены и cooldown.
    - Поправка снимает все ждущие срабатывания. Снятое освобождает токены и cooldown, как будто
      его не было. Вместе с ним снимаются частичные совпадения и окна тишины, в которые входит
      его последний токен, а копии его префикса отсчитывают задержку от поправки. Окна тишины
      копий префикса не сдвигаются: окно мерит тишину, а не задержку. Поправка закрывает и
      открытое отрицание.
    - Отрицание открывается фразой из `negations` и подавляет первое завершённое совпадение с
      шагом после фразы: срабатывание этого момента не выдаётся и не занимает токены и cooldown,
      а дальше — как при снятии поправкой. Потом отрицание закрывается. Закрывают его и фраза из
      `negation_breaks`, и срок: `max_gap` корня от фразы, а затем от каждого нового шага любого
      совпадения. Фраза отрицания при открытом отрицании только продлевает срок.
    - Маркер внутри шаблона: если токен маркера стал явным шагом совпадения (не шагом `any` и
      не `any_vac`), действие маркера ждёт. Сработает совпадение с этим шагом — маркер
      отменяется: для совпадения это обычный шаг. Умрут все такие совпадения — маркер
      применяется так, как применился бы в свой момент: поправка снимает ждущие на её момент,
      отрицание подавляет первые срабатывания после себя, разрыв закрывает отрицание с
      момента разрыва. Срабатывания, которые от этого зависят, ждут решения и по сроку не
      выдаются; копии префикса снятых отсчитывают задержку от момента решения. Пока маркер
      ждёт, `next_deadline` включает сроки снятия совпадений, а `advance` их снимает.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._defaults = rules  # стандартные задержка и пороги не меняются
        self._correcting = rules.self_correction  # режим поправок
        # сценарий → задержка выдачи; без режима поправок у всех 0: срабатывание выдаётся сразу
        self._holds = {scenario.name: self._hold_of(scenario) for scenario in rules.scenarios}
        self._held: list[Hold] = []  # ждущие срабатывания в порядке завершения
        # сценарий → момент завершения последнего выданного срабатывания: к нему возвращается
        # cooldown, если поправка снимет ждущие
        self._settled_at: dict[str, float] = {}
        # фраза-маркер → что она делает; без режима поправок маркеров нет
        self._phrases: dict[Phrase, str] = {}
        if self._correcting:
            for kind, phrases in (
                ("correction", rules.corrections),
                ("negation", rules.negations),
                ("break", rules.negation_breaks),
            ):
                self._phrases.update(dict.fromkeys(phrases, kind))
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
        self._holds[scenario.name] = self._hold_of(scenario)
        self._intercepting |= self._marker_tokens & pattern_tokens((scenario,))
        self._declared[scenario.name] = len(self._declared)
        self._min_confidences[scenario.name] = rules.confidence_threshold(scenario)
        self._vocabulary = vocabulary
        self._lookback = max(self._lookback, before_of(scenario))

    @property
    def vocabulary(self) -> frozenset[str]:
        """Текущий словарь, включая токены успешно добавленных сценариев."""
        return self._vocabulary

    @property
    def held(self) -> list[Held]:
        """Срабатывания, которые ждут выдачи (режим поправок), в порядке завершения."""
        return [
            Held(hold.scenario, hold.partial, hold.deadline, bool(hold.waits))
            for hold in self._held
        ]

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
        """Ближайший конец окна тишины или срок ждущего срабатывания: когда `advance` выдаст
        что-то без события. None — ждать нечего."""
        deadline = self._wheel.next_deadline()
        times = [hold.deadline for hold in self._held if not hold.waits]
        if self._deferred:
            # отложенный маркер решается, когда умрут совпадения: сроки снятия первых во фронтах
            for front in self._all_fronts():
                if len(front):
                    place = front.place
                    if isinstance(place, Cursor):
                        times.append(expiry(front.first().last_time, place.max_gap, True))
                    else:
                        times.append(expiry(front.first().last_time, place[0].pattern.max_gap, False))
        if times:
            first = min(times)
            deadline = first if deadline is None else min(deadline, first)
        return deadline

    def feed(self, events: Iterable[Event]) -> list[Match]:
        matches: list[Match] = []
        for event in events:
            matches.extend(self.push(event))
        return matches

    def advance(self, time: float) -> list[Match]:
        """Двигает время без события: закрывает окна тишины и выдаёт ждущие срабатывания со
        сроком раньше `time`."""
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
        self._move(event.time, f"время {event.time} токена {event.token!r}")
        misses = self._misses = []
        matches = self._close(event.time)  # окна и ждущие, истёкшие раньше события
        step = wild = Step(self._position, event)
        self._position += 1
        stepped = None
        if self._correcting:
            stepped = self._stepped = set()
            if event.token in self._intercepting:
                # токен события — маркер из шаблонов: шаг по явному токену и шаг по `any` — разные
                # объекты, маркер перехватывает только явный шаг (`has`)
                wild = Step(step.position, event)
            if self._phrases:
                self._said.append(step)
                first = self._said[0].position
                self._used = [used for used in self._used if used.position >= first]
        token = event.token
        explicit = wild is not step  # иначе все шаги события — один объект
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
        self._expire(event.time, event, keys)
        for key in keys:
            mine = wild if explicit and key != token else step
            for front in self._waiting.get(key, ()):
                cursor = front.place
                for edge in front.edges[key]:
                    self._follow(front, edge, cursor.follow(edge), mine, completed, entries)
        for key in keys:
            mine = wild if explicit and key != token else step
            for start in self._start_index.get(key, ()):
                child = Cursor(self._starts[start], 1)
                if event.score >= start.min_score:
                    # новое совпадение задержку не проверяет
                    self._enter(child, Partial.begin(mine), completed, entries)
                    if stepped is not None:
                        stepped.add(step.position)
                else:
                    misses.append(Miss("score", child, None, None, start.min_score, event.score))
        seen: set[Collector] = set()  # набор с шагами `<шаги>` и `any` стоит в индексе дважды
        for key in keys:
            for collector in self._collector_index.get(key, ()):
                if collector not in seen:
                    seen.add(collector)
                    self._collect(collector, step, wild, completed, entries)
        for place, partial in entries:
            self._store(place, partial)
        # срабатывания выдаются после обхода: из совпадений сценария выбирается самое уверенное
        if completed:
            matches += self._finish(completed, event.time)
        if stepped is not None:
            if stepped:
                # шаг родственника: совпадение с тем же первым шагом продолжает фразу
                for hold in self._held:
                    if hold.partial.first_position in stepped:
                        hold.deadline = max(hold.deadline, event.time + hold.hold)
                negation = self._negation
                if negation is not None:
                    negation.deadline = max(negation.deadline, event.time + self._defaults.max_gap)
            matches += self._settle_markers(event.time)
            if self._phrases:
                self._hear(step)

        self._recent.append(step)
        while self._recent and self._recent[0].event.time < event.time - self._lookback:
            self._recent.popleft()
        self._tick()
        return matches

    def _hear(self, step: Step) -> None:
        """Маркеры, которые кончаются событием `step`: действуют сразу, отменены или ждут."""
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
        """Маркер ждёт решения: поправка держит ждущие и открытое отрицание, разрыв — отрицание,
        отрицание открывается под вопросом."""
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
        """Стоит ли один из шагов маркера явным шагом живого совпадения."""
        partials = chain(
            (partial for front in self._all_fronts() for partial in front),
            (timer.payload.partial for timer in self._hushes),
        )
        return any(has(partial, steps) for partial in partials)

    def _witness(self, partials: list[Partial], time: float) -> None:
        """Совпадения момента, которые сработают: маркер, ставший их шагом, отменяется."""
        said = [step for step in self._said if step.event.token in self._intercepting]
        self._used += [step for step in said if any(has(partial, [step]) for partial in partials)]
        for marker in list(self._deferred):
            if any(has(partial, marker.steps) for partial in partials):
                self._settle(marker, "cancelled", time)

    def _settle_markers(self, moment: float) -> list[Match]:
        """Отложенные маркеры, у которых умерли все совпадения с их шагом, применяются в
        `moment`; ждущие, которых больше ничто не держит, выдаются, если срок уже прошёл."""
        for marker in list(self._deferred):
            if not self._alive(marker.steps):
                self._settle(marker, "applied", moment)
        now = self._last_time
        return self._release(lambda due: due < now) if self._held else []

    def _unfreeze(self, hold: Hold, wait: "Marker | Victims", moment: float) -> None:
        """Маркер больше не держит ждущее; отпущенное после срока выдаётся после `moment`."""
        hold.waits.discard(wait)
        if not hold.waits:
            hold.deadline = max(hold.deadline, moment)

    def _settle(self, marker: Marker, state: str, moment: float) -> None:
        marker.state = state
        self._deferred.remove(marker)
        if marker.kind == "correction":
            frozen = [hold for hold in marker.holds if hold in self._held]
            if state == "applied":
                self._cancel(frozen, moment, "corrected")
            else:
                for hold in frozen:
                    self._unfreeze(hold, marker, moment)
        negation = self._negation
        if negation is not None:
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
        for group in list(self._groups):
            self._judge(group, moment)

    def _judge(self, group: Victims, moment: float) -> None:
        """Подавить срабатывания группы или отпустить их, если маркеры уже решены."""
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
        return [
            *self._fronts.values(),
            *(front for collector in self._collectors for front in collector.states.values()),
        ]

    def _heard(self) -> dict[str, list[Step]]:
        """Фразы-маркеры, которые кончаются последним событием: что делают → шаги их токенов."""
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
        """Снимает ждущие срабатывания: токены и cooldown освобождаются, как будто их не было.

        `need` отказа — срок выдачи снятого, у подавленного отрицанием — момент отрицания.
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
        for scenario in {hold.scenario for hold in holds}:
            times = [hold.time for hold in self._held if hold.scenario == scenario]
            if scenario in self._settled_at:
                times.append(self._settled_at[scenario])
            if times:
                self._fired_at[scenario] = max(times)
            else:
                del self._fired_at[scenario]
        self._withdraw([hold.partial for hold in holds], time, reason)

    def _withdraw(self, partials: list[Partial], time: float, reason: str) -> None:
        """Чистка после снятых срабатываний `partials` в момент `time`.

        Совпадения и окна тишины, в которые входит последний токен снятого, снимаются: иначе
        «открой шторы, ой, …, разогрей чайник» собрало бы «утро». Копии префикса снятого
        отсчитывают задержку от `time`; чужие совпадения в тех же местах не меняются.
        """
        lasts = {partial.tail.step.position for partial in partials}
        low = min(lasts)
        prefixes: dict[int, set[tuple[int, ...]]] = {}  # первая позиция → префиксы снятых
        for partial in partials:
            positions = partial.positions()
            prefixes.setdefault(partial.first_position, set()).update(
                tuple(positions[:length]) for length in range(1, len(positions))
            )

        def shares(partial: Partial) -> bool:
            steps = takewhile(lambda step: step.position >= low, partial.backwards())
            return any(step.position in lasts for step in steps)

        def copies(partial: Partial) -> bool:
            own = prefixes.get(partial.first_position)
            return own is not None and tuple(partial.positions()) in own

        for front in self._all_fronts():
            self._rework(front, shares, copies, time, reason)
        for timer in [timer for timer in self._hushes if shares(timer.payload.partial)]:
            window, partial = timer.payload
            self._wheel.cancel(timer)
            self._unlist(timer)
            self._misses.append(Miss(reason, window, partial, None, time, time))

    def _rework(
        self,
        front: Front,
        drop: Callable[[Partial], bool],
        restart: Callable[[Partial], bool],
        time: float,
        reason: str,
    ) -> None:
        """Снимает из фронта совпадения `drop`, а совпадениям `restart` переносит отсчёт на
        `time`. Опустевший фронт остаётся на месте до своей записи в очереди сроков."""
        kept: list[Partial] = []
        moved: list[Partial] = []
        for partial in front:
            if drop(partial):
                place = front.place
                if not isinstance(place, Cursor):
                    place = place[0].place(place[1])
                self._misses.append(Miss(reason, place, partial, None, time, time))
            elif restart(partial) and partial.last_time < time:
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
        self._active_count += front.add(partial)
        if not front.queued:
            self._schedule(front)

    def _schedule(self, front: Front) -> None:
        """Ставит непустой фронт в очередь по сроку его самого старого кандидата."""
        place = front.place
        gap = place.max_gap if isinstance(place, Cursor) else place[0].pattern.max_gap
        due = due_time(front.first().last_time, gap)
        heapq.heappush(self._deadlines, (due, next(self._order), front))
        front.queued = True

    def _expire(self, now: float, event: Event | None, keys: tuple[str, ...]) -> float | None:
        """Снимает кандидатов, срок которых к `now` истёк: во фронте это голова.

        Без события (`advance` в режиме поправок) возвращает самый поздний срок снятых: тогда
        умерло последнее из них; None — никого не сняли.
        """
        latest: float | None = None
        due: list[Front] = []
        while self._deadlines and self._deadlines[0][0] <= now:
            due.append(heapq.heappop(self._deadlines)[2])
        for front in due:
            front.queued = False
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
                    if event is None:
                        latest = max(expiry(last, gap, True), latest or -math.inf)
            else:
                collector, filled = place
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
                    self._misses.append(
                        Miss(reason, collector.place(filled), partial, None, deadline, now)
                    )
                    self._active_count -= 1
                    if event is None:
                        latest = max(deadline, latest or -math.inf)
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
        return latest

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
        if self._stepped is not None:
            self._stepped.update(items[index].first_position for index in range(fresh, len(items)))
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
        wild: Step,
        completed: dict[str, list[Partial]],
        entries: list,
    ) -> None:
        """Событие в шаблоне `any`: закрывает открытые шаги у его состояний и начинает новые.

        `wild` — шаг события для шагов набора `any` и `any_vac` (`Engine.push`).
        """
        event = step.event
        pattern = collector.pattern
        length = len(pattern.slots)
        threshold = pattern.min_confidence

        def pick(index: int) -> Step:
            return step if pattern.slots[index].token == event.token else wild

        def gather(target: int, items: list[Partial], start: int, step: Step) -> None:
            """Продолжения `items[start:]` событием входят в состояние `target`."""
            if self._stepped is not None:
                self._stepped.update(items[index].first_position for index in range(start, len(items)))
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
                gather(filled | 1 << index, front.items, front.head, pick(index))
            for index in low:
                need = pattern.slots[index].min_score
                place = collector.place(filled)
                self._misses += [
                    Miss("score", place, partial, None, need, event.score) for partial in front
                ]
        # начать набор может любой его шаг
        free, low = collector.closes(0, groups, event.score)
        if free and self._stepped is not None:
            self._stepped.add(step.position)
        for index in free:
            partial = Partial.begin(pick(index))
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
        """Окна тишины со сроком раньше `now`: тишина выдержана, совпадение идёт за ребро.

        Ждущие срабатывания выдаются по порядку сроков вперемешку с окнами: срабатывание,
        которое завершило окно, может истечь до следующего окна.
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
        if self._negation is not None and now > self._negation.deadline:
            self._negation = None

    def _release(self, due: Callable[[float], bool]) -> list[Match]:
        """Выдаёт ждущие срабатывания, срок которых прошёл (`due`): по срокам, один срок — один
        момент, внутри момента — как у `_finish`. Ждущие решения маркера не выдаются."""
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
        chosen = []  # (сценарий, его совпадение): прошли порог, повтор и cooldown
        for scenario, partials in completed.items():
            best = self._choose(scenario, partials, time)
            if best is not None:
                chosen.append((scenario, best))
        if self._intercepting and chosen:
            self._witness([best for _, best in chosen], time)
        matches = []
        suppressed: list[Partial] = []  # подавленные отрицанием: все срабатывания момента
        victims: list[Hold] = []  # подавит отрицание, если отложенные маркеры решатся так
        for scenario, best in chosen:
            match = self._fire(scenario, best, time, suppressed, victims)
            if match is not None:
                matches.append(match)
        if suppressed or victims:
            negation = self._negation
            self._negation = None
            if victims:
                group = Victims(negation, victims)
                for hold in victims:
                    hold.waits.add(group)
                self._groups.append(group)
            else:
                self._withdraw(suppressed, time, "negated")
        matches.sort(key=lambda match: (match.steps[0].position, self._declared[match.scenario]))
        return matches

    def _choose(self, scenario: str, partials: list[Partial], time: float) -> Partial | None:
        """Совпадение, которым сработает сценарий: самое уверенное из прошедших порог и повтор,
        если сценарий не молчит по cooldown."""
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
        return best

    def _fire(
        self,
        scenario: str,
        best: Partial,
        time: float,
        suppressed: list[Partial],
        victims: list[Hold],
    ) -> Match | None:
        """Срабатывание: выдаётся сразу, ждёт выдачи или подавлено отрицанием."""
        hold = self._holds[scenario]
        negation = self._negation
        tentative = False  # отрицание с отложенным маркером или разрывом
        if hold > 0 and negation is not None and best.tail.step.position > negation.position:
            if negation.marker is None and not negation.breaks:
                self._misses.append(Miss("negated", None, best, scenario, negation.time, time))
                suppressed.append(best)
                return None
            tentative = True
        self._taken[scenario].update(step.position for step in best.backwards())
        self._taken_count += best.count  # позиции новые: проверено выше
        self._fired_at[scenario] = time
        if hold > 0:
            held = Hold(scenario, best, time, hold)
            self._held.append(held)
            self._active_count += 1
            if tentative:
                victims.append(held)
            return None
        if self._correcting:
            self._settled_at[scenario] = time
        return Match(scenario, best.steps, best.confidence)

    def _hold_of(self, scenario: Scenario) -> float:
        return self._defaults.hold_of(scenario) if self._correcting else 0.0

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
