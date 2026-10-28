"""Проход потока токенов по графу сценариев, по шаблонам `any` и по окнам тишины."""

import math
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import NamedTuple

from rule_engine.domain.graph import (
    Cursor,
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


def keep(partials: list[Partial], partial: Partial) -> None:
    """Добавляет совпадение к несравнимым, если его никто не вытесняет, и убирает вытесненные им."""
    if any(dominates(other, partial) for other in partials):
        return
    partials[:] = [other for other in partials if not dominates(partial, other)]
    partials.append(partial)


class Collector:
    """Шаблон `any` в движке: закрытые шаги — битовая маска, одинаковые шаги — одна группа."""

    __slots__ = ("pattern", "groups", "full")

    def __init__(self, pattern: Unordered) -> None:
        self.pattern = pattern
        self.groups = pattern.groups()  # токен шага → (порог, маска номеров) групп
        self.full = (1 << len(pattern.slots)) - 1  # маска собранного шаблона

    def free(self, filled: int, token: str, score: float, vocabulary: frozenset[str]) -> list[int]:
        """`Unordered.free` на масках: номера шагов, которые событие может закрыть."""
        free = []
        for key in fitting(token, vocabulary):
            for min_score, mask in self.groups.get(key, ()):
                open_ = mask & ~filled
                if open_ and score >= min_score:
                    free.append((open_ & -open_).bit_length() - 1)
        free.sort()
        return free

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
        self._unordered = build_unordered(rules)
        self._unordered_owners = {
            (tuple(sorted(pattern.slots)), pattern.max_gap): pattern.scenario
            for pattern in self._unordered
        }
        self._collectors = [Collector(pattern) for pattern in self._unordered]
        self._active: dict[Cursor, list[Partial]] = {}  # позиция radix trie → кандидаты
        # (шаблон `any`, маска закрытых шагов) → несравнимые совпадения в этом состоянии
        self._sets: dict[tuple[Collector, int], list[Partial]] = {}
        self._wheel: TimerWheel[Hush] = TimerWheel()  # сроки окон тишины
        self._hushes: dict[Timer[Hush], None] = {}  # открытые окна в порядке открытия
        self._lookback = longest_before(self._starts)  # секунды: самое длинное окно назад
        self._recent: deque[Step] = deque()  # события не старше `_lookback`: для тишины до шага
        self._taken: dict[str, set[int]] = {}  # сценарий → позиции токенов его выданных срабатываний
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
        rules = RuleSet(
            vocabulary,
            (scenario,),
            self._defaults.max_gap,
            self._defaults.min_score,
            self._defaults.min_confidence,
        )
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

                self._active = {
                    relocate(place): partials for place, partials in self._active.items()
                }
                self._misses = [miss._replace(place=relocate(miss.place)) for miss in self._misses]
            # Концы старых рёбер сохраняют identity и silences: Window и Timer не меняются.
        else:
            known = len(self._unordered)
            insert_unordered(self._unordered, self._unordered_owners, rules)
            self._collectors += [Collector(pattern) for pattern in self._unordered[known:]]
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
        return (
            sum(len(partials) for partials in self._active.values())
            + sum(len(partials) for partials in self._sets.values())
            + len(self._hushes)
        )

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
        # В `_active` только позиции с токеновым продолжением, в том числе внутри сегмента.
        waiting = []
        for cursor, partials in self._active.items():
            gap = max(edge.max_gap for edge, _ in cursor.transitions())
            waiting.extend(
                Waiting(cursor, partial, partial.last_time + gap) for partial in partials
            )
        waiting += [
            Waiting(collector.place(filled), partial, partial.last_time + collector.pattern.max_gap)
            for (collector, filled), partials in self._sets.items()
            for partial in partials
        ]
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
        self._forget()
        return matches

    def push(self, event: Event) -> list[Match]:
        self._move(event.time, f"время {event.time} токена {event.token!r}")
        misses = self._misses = []
        matches = self._close(event.time)  # окна, истёкшие раньше события
        step = Step(self._position, event)
        self._position += 1

        active: dict[Cursor, list[Partial]] = {}
        sets: dict[tuple[Collector, int], list[Partial]] = {}
        completed: dict[str, list[Partial]] = {}  # сценарий → совпадения, завершённые этим событием

        def collect(collector: Collector, filled: int, partial: Partial) -> None:
            pattern = collector.pattern
            length = len(pattern.slots)
            if filled == collector.full:
                completed.setdefault(pattern.scenario, []).append(partial)
            elif partial.bound(length) >= pattern.min_confidence:  # ранний отсев
                keep(sets.setdefault((collector, filled), []), partial)
            else:
                misses.append(
                    Miss(
                        "pruned",
                        collector.place(filled),
                        partial,
                        None,
                        pattern.min_confidence,
                        partial.bound(length),
                    )
                )

        def low_slots(collector: Collector, filled: int) -> list[int]:
            """Открытые шаги набора с токеном события, порог которых событие не прошло."""
            return [
                index
                for index in collector.free(filled, event.token, 1.0, self._vocabulary)
                if event.score < collector.pattern.slots[index].min_score
            ]

        # нарушенная тишина: истёкшие окна закрыты выше, поэтому оставшиеся ещё идут
        for timer in list(self._hushes):
            window, partial = timer.payload
            if self._breaks(window.edge, event):
                self._wheel.cancel(timer)
                del self._hushes[timer]
                misses.append(Miss("silence", window, partial, None, timer.deadline, event.time))
        for node, partials in self._active.items():
            transitions = tuple(node.transitions())
            max_gap = max(edge.max_gap for edge, _ in transitions)
            for partial in partials:
                # срок считается от последнего шага совпадения, а не от последнего токена потока
                last = partial.last_time
                elapsed = event.time - last
                # сроки всех рёбер из вершины истекли: совпадение снято
                expired = elapsed > max_gap
                if not expired:
                    # копия остаётся в вершине и ждёт другого продолжения
                    keep(active.setdefault(node, []), partial)
                matched = False  # есть ребро с токеном события: снятие объясняет опоздание
                for edge, child in transitions:
                    if not fits(edge.token, event.token, self._vocabulary):
                        continue
                    matched = True
                    if elapsed > edge.max_gap:
                        misses.append(
                            Miss("late", child, partial, None, last + edge.max_gap, event.time)
                        )
                    elif event.score < edge.min_score:
                        misses.append(
                            Miss("score", child, partial, None, edge.min_score, event.score)
                        )
                    else:
                        self._enter(child, partial.extend(step), active, completed)
                if expired and not matched:
                    deadline = last + max_gap
                    misses.append(Miss("expired", node, partial, None, deadline, event.time))
        for start, child in self._starts.items():
            if not fits(start.token, event.token, self._vocabulary):
                continue
            if event.score >= start.min_score:
                # новое совпадение задержку не проверяет
                self._enter(Cursor(child, 1), Partial.begin(step), active, completed)
            else:
                misses.append(Miss("score", Cursor(child, 1), None, None, start.min_score, event.score))
        for (collector, filled), partials in self._sets.items():
            pattern = collector.pattern
            for partial in partials:
                deadline = partial.last_time + pattern.max_gap
                if event.time > deadline:
                    # задержка у открытых шагов одна, и она истекла: совпадение снято
                    late = collector.free(filled, event.token, 1.0, self._vocabulary)
                    reason = "late" if late else "expired"
                    misses.append(
                        Miss(reason, collector.place(filled), partial, None, deadline, event.time)
                    )
                    continue
                keep(sets.setdefault((collector, filled), []), partial)  # копия ждёт другого шага
                for index in collector.free(filled, event.token, event.score, self._vocabulary):
                    collect(collector, filled | 1 << index, partial.extend(step))
                for index in low_slots(collector, filled):
                    threshold = pattern.slots[index].min_score
                    place = collector.place(filled)
                    misses.append(Miss("score", place, partial, None, threshold, event.score))
        for collector in self._collectors:
            # начать набор может любой его шаг
            for index in collector.free(0, event.token, event.score, self._vocabulary):
                collect(collector, 1 << index, Partial.begin(step))
            for index in low_slots(collector, 0):
                threshold = collector.pattern.slots[index].min_score
                misses.append(
                    Miss("score", collector.place(0), None, None, threshold, event.score)
                )
        # срабатывания выдаются после обхода: из совпадений сценария выбирается самое уверенное
        matches += self._finish(completed, event.time)

        self._active = active
        self._sets = sets
        self._recent.append(step)
        while self._recent and self._recent[0].event.time < event.time - self._lookback:
            self._recent.popleft()
        self._forget()
        return matches

    def _move(self, time: float, what: str) -> None:
        if self._last_time is not None and time < self._last_time:
            raise ValueError(f"{what} раньше предыдущего {self._last_time}")
        self._last_time = time

    def _close(self, now: float) -> list[Match]:
        """Окна тишины со сроком раньше `now`: тишина выдержана, совпадение идёт за ребро."""
        matches: list[Match] = []
        # окна с одним сроком закрываются в один момент, как совпадения одного события
        fired = self._wheel.advance(now)
        for deadline, timers in groupby(fired, key=lambda timer: timer.deadline):
            completed: dict[str, list[Partial]] = {}
            for timer in timers:
                del self._hushes[timer]
                window, partial = timer.payload
                # после тишины не бывает тишины (rules.py), поэтому окна здесь не открываются
                self._enter(window.child, partial._replace(last_time=deadline), self._active, completed)
            matches += self._finish(completed, deadline)
        return matches

    def _enter(
        self,
        node: Cursor,
        partial: Partial,
        active: dict[Cursor, list[Partial]],
        completed: dict[str, list[Partial]],
    ) -> None:
        if node.scenario is not None:
            completed.setdefault(node.scenario, []).append(partial)
        # ранний отсев; у вершины без продолжений `ahead` пуст, и совпадение в ней не остаётся
        if node.ahead.reaches(partial.product):
            if node.offset < len(node.node.segment) or node.node.children:
                keep(active.setdefault(node, []), partial)
            for edge in node.silences():
                self._open(Window(node.node, edge), partial)
        elif node.ahead:
            # ближе всего к порогу — шаблон с наибольшим запасом
            length, threshold = max(node.ahead, key=lambda item: partial.bound(item[0]) - item[1])
            self._misses.append(
                Miss("pruned", node, partial, None, threshold, partial.bound(length))
            )

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
        self._fired_at[scenario] = time
        return Match(scenario, best.steps, best.confidence)

    def _forget(self) -> None:
        """Новые срабатывания состоят из позиций не раньше начала самого старого активного
        совпадения, более ранние можно забыть."""
        # один `min` по всем местам: `default` второго `min` не учёл бы совпадения в вершинах и
        # наборах, когда открыто хоть одно окно
        oldest = min(
            (
                *(
                    partial.first_position
                    for partials in (*self._active.values(), *self._sets.values())
                    for partial in partials
                ),
                *(timer.payload.partial.first_position for timer in self._hushes),
            ),
            default=self._position,
        )
        for taken in self._taken.values():
            taken.difference_update([position for position in taken if position < oldest])
