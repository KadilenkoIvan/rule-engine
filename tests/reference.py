"""Эталон для дифференциальной проверки движка: проход main с полным обходом.

На каждом событии обходятся все кандидаты в вершинах, все наборы `any` и все окна тишины, а
активные места собираются заново. Индексов, очередей и счётчиков нет, поэтому эталон не
делит с движком ни одной оптимизации прохода. Общие с движком только правила, граф,
колесо таймеров и типы данных.

Семантика — как у движка, включая решения о ничьей и порядке выдачи (PLAN.md, Р18).
"""

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import replace
from itertools import groupby

from rule_engine.domain.engine import (
    Event,
    Hush,
    Match,
    Miss,
    Partial,
    Place,
    Step,
    Waiting,
    Window,
)
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
    parse_scenario,
    step_tokens,
)
from rule_engine.domain.timer_wheel import Timer, TimerWheel
from rule_engine.domain.unordered import Unordered, build_unordered, insert_unordered


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


class ReferenceEngine:
    """Тот же контракт, что у `Engine`: правила прохода описаны там."""

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
        self._active: dict[Cursor, list[Partial]] = {}  # позиция radix trie → кандидаты
        # (шаблон `any`, номера закрытых шагов) → несравнимые совпадения в этом состоянии
        self._sets: dict[tuple[Unordered, frozenset[int]], list[Partial]] = {}
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
        rules = replace(self._defaults, vocabulary=vocabulary, scenarios=(scenario,))
        if scenario.order == "sequence":
            for old, prefix, cut in insert_scenario(self._starts, rules, scenario).splits:

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
            insert_unordered(self._unordered, self._unordered_owners, rules)
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
            Waiting(key, partial, partial.last_time + key[0].max_gap)
            for key, partials in self._sets.items()
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
        sets: dict[tuple[Unordered, frozenset[int]], list[Partial]] = {}
        completed: dict[str, list[Partial]] = {}  # сценарий → совпадения, завершённые этим событием

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
                for index in pattern.free(filled, event.token, 1.0, self._vocabulary)
                if event.score < pattern.slots[index].min_score
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
        for (pattern, filled), partials in self._sets.items():
            for partial in partials:
                deadline = partial.last_time + pattern.max_gap
                if event.time > deadline:
                    # задержка у открытых шагов одна, и она истекла: совпадение снято
                    fitting = pattern.free(filled, event.token, 1.0, self._vocabulary)
                    reason = "late" if fitting else "expired"
                    misses.append(
                        Miss(reason, (pattern, filled), partial, None, deadline, event.time)
                    )
                    continue
                keep(sets.setdefault((pattern, filled), []), partial)  # копия ждёт другого шага
                for index in pattern.free(filled, event.token, event.score, self._vocabulary):
                    collect(pattern, filled | {index}, partial.extend(step))
                for index in low_slots(pattern, filled):
                    threshold = pattern.slots[index].min_score
                    misses.append(
                        Miss("score", (pattern, filled), partial, None, threshold, event.score)
                    )
        for pattern in self._unordered:
            # начать набор может любой его шаг
            for index in pattern.free(frozenset(), event.token, event.score, self._vocabulary):
                collect(pattern, frozenset({index}), Partial.begin(step))
            for index in low_slots(pattern, frozenset()):
                threshold = pattern.slots[index].min_score
                misses.append(
                    Miss("score", (pattern, frozenset()), None, None, threshold, event.score)
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
        if any(partial.bound(length) >= threshold for length, threshold in node.ahead):
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
