"""Проход потока токенов по графу сценариев и по шаблонам `any`."""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import NamedTuple

from rule_engine.domain.graph import Cursor, Node, Start, build_graph, insert_scenario
from rule_engine.domain.rules import RuleSet, Scenario, parse_scenario
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
    steps: tuple[Step, ...]
    confidence: float  # общая уверенность: среднее геометрическое score шагов


@dataclass(frozen=True, slots=True)
class History:
    """Неизменяемое звено: общие префиксы кандидатов не копируются."""

    step: Step
    previous: "History | None" = None


class Partial(NamedTuple):
    """История по ссылке и метаданные для O(1) продления и сравнения."""

    tail: History
    product: float
    count: int
    first_position: int

    @classmethod
    def begin(cls, step: Step) -> "Partial":
        return cls(History(step), step.event.score, 1, step.position)

    def backwards(self) -> Iterator[Step]:
        link = self.tail
        while link is not None:
            yield link.step
            link = link.previous

    @property
    def steps(self) -> tuple[Step, ...]:
        """Материализация нужна только для результата и представления интерфейса."""
        steps = list(self.backwards())
        steps.reverse()
        return tuple(steps)

    @property
    def last(self) -> Step:
        return self.tail.step

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
        )


class Waiting(NamedTuple):
    """Частичное совпадение в снимке движка: где оно стоит и до какого момента ждёт."""

    place: Cursor | tuple[Unordered, frozenset[int]]
    partial: Partial
    until: float  # секунды: срок продолжения


class Miss(NamedTuple):
    """Почему событие не продвинуло совпадение или не дало сработать сценарию.

    - `"score"`: score токена `got` ниже порога шага `need`. `place` — куда не пустило: вершина
      за ребром или состояние набора `any`.
    - `"late"`: токен подошёл бы, но пришёл в `got`, а ждали до `need`. `place` — как у `"score"`.
    - `"expired"`: совпадение снято: ждали до `need`, а токен в `got` ему не подходил. `place` —
      где оно стояло.
    - `"pruned"`: совпадение отсеяно: даже при score 1.0 у оставшихся шагов уверенность не выше
      `got`, а порог `need`. `place` — куда оно вошло.
    - `"confidence"`: шаблон сценария собран, но общая уверенность `got` ниже порога `need`.
    - `"repeat"`: шаблон собран, но его токен уже вошёл в выданное срабатывание сценария.
    - `"cooldown"`: шаблон собран, но сценарий молчит до `need`.

    `partial` — совпадение, которое не продвинулось; `None` — новое совпадение не началось.
    У трёх последних причин есть `scenario`, а `place` — `None`.
    """

    reason: str
    place: Cursor | tuple[Unordered, frozenset[int]] | None
    partial: Partial | None
    scenario: str | None
    need: float
    got: float


def dominates(a: Partial, b: Partial) -> bool:
    """`a` вытесняет `b` из вершины: последний шаг у `a` не раньше, а произведение не меньше.

    Тогда любое продолжение `b` есть и у `a`, со сроком не раньше и уверенностью не ниже. Если
    равны и последний шаг, и произведение, остаётся начавшееся позже.
    """
    if (a.last.position, a.product) == (b.last.position, b.product):
        return a.first_position >= b.first_position
    return a.last.position >= b.last.position and a.product >= b.product


def keep(partials: list[Partial], partial: Partial) -> None:
    """Добавляет совпадение к несравнимым, если его никто не вытесняет, и убирает вытесненные им."""
    if any(dominates(other, partial) for other in partials):
        return
    partials[:] = [other for other in partials if not dominates(partial, other)]
    partials.append(partial)


class Engine:
    """Частичные совпадения хранятся между вызовами, поэтому поток можно подавать порциями.

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
    - Если на одном событии завершились несколько совпадений одного сценария, выдаётся не
      больше одного: последний токен у них общий. Выдаётся самое уверенное, а при равной
      уверенности — начавшееся позже.
    - Срабатывание не выдаётся, если у него есть общий токен с выданным срабатыванием того же
      сценария (токены сравниваются по позиции в потоке) или если с выданного срабатывания
      этого сценария прошло меньше его `cooldown`. Невыданное срабатывание не занимает токены
      и не продлевает cooldown, а совпадение живёт дальше.
    - Время событий не убывает: событие раньше предыдущего — ошибка.
    - Неразветвлённые цепочки сжаты в radix-рёбра; кандидат занимает позицию внутри ребра.
      История хранится связно: продление создаёт одно звено, не копируя общий префикс.
    - `add_scenario` дописывает пути, при необходимости расщепляя radix-рёбра. Активные
      совпадения сохраняют позиции и могут продолжаться по добавленным веткам.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._defaults = rules  # стандартные задержка и пороги не меняются
        self._vocabulary = rules.vocabulary
        self._cooldowns = {scenario.name: scenario.cooldown for scenario in rules.scenarios}
        self._min_confidences = {
            scenario.name: rules.confidence_threshold(scenario) for scenario in rules.scenarios
        }
        self._starts = build_graph(rules)
        self._unordered = build_unordered(rules)
        self._unordered_owners = {
            (tuple(sorted(pattern.slots)), pattern.max_gap): pattern.scenario
            for pattern in self._unordered
        }
        self._active: dict[Cursor, list[Partial]] = {}  # позиция в radix trie → кандидаты
        # (шаблон `any`, номера закрытых шагов) → несравнимые совпадения в этом состоянии
        self._sets: dict[tuple[Unordered, frozenset[int]], list[Partial]] = {}
        self._taken: dict[str, set[int]] = {}  # сценарий → позиции токенов его выданных срабатываний
        self._fired_at: dict[str, float] = {}  # сценарий → время его последнего выданного срабатывания
        self._position = 0
        self._last_time: float | None = None  # None, пока не пришло ни одного события
        self._misses: list[Miss] = []  # отказы последнего события

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
            step.token for pattern in scenario.patterns for step in pattern
        } - self._vocabulary
        vocabulary = self._vocabulary | new_tokens if new_tokens else self._vocabulary
        rules = RuleSet(
            vocabulary,
            (scenario,),
            self._defaults.max_gap,
            self._defaults.min_score,
            self._defaults.min_confidence,
        )
        if scenario.order == "sequence":
            splits = insert_scenario(self._starts, rules, scenario)
            # При расщеплении сжатого ребра курсоры сохраняют логическую позицию.
            for old, prefix, cut in splits:

                def relocate(place: Cursor | tuple[Unordered, frozenset[int]] | None):
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
        else:
            insert_unordered(self._unordered, self._unordered_owners, rules)
        self._cooldowns[scenario.name] = scenario.cooldown
        self._min_confidences[scenario.name] = rules.confidence_threshold(scenario)
        self._vocabulary = vocabulary

    @property
    def vocabulary(self) -> frozenset[str]:
        """Текущий словарь, включая токены успешно добавленных сценариев."""
        return self._vocabulary

    @property
    def active_count(self) -> int:
        return sum(len(partials) for partials in self._active.values()) + sum(
            len(partials) for partials in self._sets.values()
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
    def misses(self) -> list[Miss]:
        """Почему последнее событие что-то не продвинуло или не дало сработать сценарию."""
        return self._misses

    def snapshot(self) -> list[Waiting]:
        """Все частичные совпадения: в вершинах бора и в наборах `any`."""
        # В `_active` только позиции с продолжениями; внутри ребра следующий шаг один.
        waiting = []
        for cursor, partials in self._active.items():
            gap = max(edge.max_gap for edge, _ in cursor.transitions())
            waiting.extend(
                Waiting(cursor, partial, partial.last.event.time + gap)
                for partial in partials
            )
        waiting += [
            Waiting(key, partial, partial.last.event.time + key[0].max_gap)
            for key, partials in self._sets.items()
            for partial in partials
        ]
        return waiting

    def feed(self, events: Iterable[Event]) -> list[Match]:
        matches: list[Match] = []
        for event in events:
            matches.extend(self.push(event))
        return matches

    def push(self, event: Event) -> list[Match]:
        if self._last_time is not None and event.time < self._last_time:
            raise ValueError(
                f"время {event.time} токена {event.token!r} раньше предыдущего {self._last_time}"
            )
        self._last_time = event.time
        step = Step(self._position, event)
        self._position += 1

        matches: list[Match] = []
        active: dict[Cursor, list[Partial]] = {}
        sets: dict[tuple[Unordered, frozenset[int]], list[Partial]] = {}
        completed: dict[str, list[Partial]] = {}  # сценарий → совпадения, завершённые этим событием
        misses: list[Miss] = []

        def enter(node: Cursor, partial: Partial) -> None:
            if node.scenario is not None:
                completed.setdefault(node.scenario, []).append(partial)
            # ранний отсев; у вершины без продолжений `ahead` пуст, и совпадение в ней не остаётся
            if any(partial.bound(length) >= threshold for length, threshold in node.ahead):
                keep(active.setdefault(node, []), partial)
            elif node.ahead:
                # ближе всего к порогу — шаблон с наибольшим запасом
                length, threshold = max(
                    node.ahead, key=lambda item: partial.bound(item[0]) - item[1]
                )
                misses.append(Miss("pruned", node, partial, None, threshold, partial.bound(length)))

        def collect(pattern: Unordered, filled: frozenset[int], partial: Partial) -> None:
            length = len(pattern.slots)
            if len(filled) == length:
                completed.setdefault(pattern.scenario, []).append(partial)
            elif partial.bound(length) >= pattern.min_confidence:  # ранний отсев
                keep(sets.setdefault((pattern, filled), []), partial)
            else:
                misses.append(
                    Miss(
                        "pruned",
                        (pattern, filled),
                        partial,
                        None,
                        pattern.min_confidence,
                        partial.bound(length),
                    )
                )

        def low_slots(pattern: Unordered, filled: frozenset[int]) -> list[int]:
            """Открытые шаги набора с токеном события, порог которых событие не прошло."""
            return [
                index
                for index in pattern.free(filled, event.token, 1.0)
                if event.score < pattern.slots[index].min_score
            ]

        def fire(scenario: str, partials: list[Partial]) -> None:
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
                        misses.append(
                            Miss("confidence", None, partial, scenario, threshold, confidence)
                        )
                    else:
                        misses.append(Miss("repeat", None, partial, scenario, 0.0, 0.0))
                return
            best = max(eligible, key=lambda partial: (partial.confidence, partial.first_position))
            fired_at = self._fired_at.get(scenario)
            if fired_at is not None and event.time - fired_at < self._cooldowns[scenario]:
                until = fired_at + self._cooldowns[scenario]
                misses.append(Miss("cooldown", None, best, scenario, until, event.time))
                return
            taken.update(step.position for step in best.backwards())
            self._fired_at[scenario] = event.time
            matches.append(Match(scenario, best.steps, best.confidence))

        for node, partials in self._active.items():
            transitions = tuple(node.transitions())
            max_gap = max(edge.max_gap for edge, _ in transitions)
            for partial in partials:
                # срок считается от последнего шага совпадения, а не от последнего токена потока
                last = partial.last.event.time
                elapsed = event.time - last
                # сроки всех рёбер из вершины истекли: совпадение снято
                expired = elapsed > max_gap
                if not expired:
                    # копия остаётся в вершине и ждёт другого продолжения
                    keep(active.setdefault(node, []), partial)
                fits = False  # есть ребро с токеном события: снятие объясняет опоздание
                for edge, child in transitions:
                    if edge.token != event.token:
                        continue
                    fits = True
                    if elapsed > edge.max_gap:
                        misses.append(
                            Miss("late", child, partial, None, last + edge.max_gap, event.time)
                        )
                    elif event.score < edge.min_score:
                        misses.append(
                            Miss("score", child, partial, None, edge.min_score, event.score)
                        )
                    else:
                        enter(child, partial.extend(step))
                if expired and not fits:
                    deadline = last + max_gap
                    misses.append(Miss("expired", node, partial, None, deadline, event.time))
        for start, child in self._starts.items():
            if start.token != event.token:
                continue
            if event.score >= start.min_score:
                enter(Cursor(child, 1), Partial.begin(step))
            else:
                misses.append(Miss("score", Cursor(child, 1), None, None, start.min_score, event.score))
        for (pattern, filled), partials in self._sets.items():
            for partial in partials:
                deadline = partial.last.event.time + pattern.max_gap
                if event.time > deadline:
                    # задержка у открытых шагов одна, и она истекла: совпадение снято
                    reason = "late" if pattern.free(filled, event.token, 1.0) else "expired"
                    misses.append(
                        Miss(reason, (pattern, filled), partial, None, deadline, event.time)
                    )
                    continue
                keep(sets.setdefault((pattern, filled), []), partial)  # копия ждёт другого шага
                for index in pattern.free(filled, event.token, event.score):
                    collect(pattern, filled | {index}, partial.extend(step))
                for index in low_slots(pattern, filled):
                    threshold = pattern.slots[index].min_score
                    misses.append(
                        Miss("score", (pattern, filled), partial, None, threshold, event.score)
                    )
        for pattern in self._unordered:
            # начать набор может любой его шаг
            for index in pattern.free(frozenset(), event.token, event.score):
                collect(pattern, frozenset({index}), Partial.begin(step))
            for index in low_slots(pattern, frozenset()):
                threshold = pattern.slots[index].min_score
                misses.append(
                    Miss("score", (pattern, frozenset()), None, None, threshold, event.score)
                )
        # срабатывания выдаются после обхода: из совпадений сценария выбирается самое уверенное
        for scenario, partials in completed.items():
            fire(scenario, partials)

        self._active = active
        self._sets = sets
        self._misses = misses
        # новые срабатывания состоят из позиций не раньше начала самого старого активного
        # совпадения, более ранние можно забыть
        oldest = min(
            (
                partial.first_position
                for partials in (*active.values(), *sets.values())
                for partial in partials
            ),
            default=self._position,
        )
        for taken in self._taken.values():
            taken.difference_update([position for position in taken if position < oldest])
        return matches
