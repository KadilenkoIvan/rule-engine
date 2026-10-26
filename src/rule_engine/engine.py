"""Проход потока токенов по графу сценариев и по шаблонам `any`."""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import NamedTuple

from rule_engine.graph import Node, Silence, build_graph
from rule_engine.rules import RuleSet
from rule_engine.unordered import Unordered, build_unordered


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


class Quiet(NamedTuple):
    """Выдержанный шаг-тишина: до `until` запрещённых токенов не было."""

    until: float  # секунды: конец окна тишины


@dataclass(frozen=True)
class Match:
    scenario: str
    steps: tuple[Step | Quiet, ...]
    confidence: float  # общая уверенность: среднее геометрическое score шагов-событий


class Partial(NamedTuple):
    """Частичное совпадение: его шаги, произведение их score и время, от которого считается
    задержка следующего шага."""

    steps: tuple[Step | Quiet, ...]
    product: float
    time: float  # время последнего шага; после тишины — конец её окна

    @property
    def events(self) -> list[Step]:
        """Шаги-события: тишина не входит ни в уверенность, ни в правило повторов."""
        return [step for step in self.steps if isinstance(step, Step)]

    @property
    def confidence(self) -> float:
        """Среднее геометрическое score шагов-событий."""
        return self.product ** (1 / len(self.events))

    def bound(self, length: int) -> float:
        """Уверенность в шаблоне длины `length`, если у оставшихся шагов score 1.0: выше не бывает."""
        return self.product ** (1 / length)

    def extend(self, step: Step) -> "Partial":
        return Partial(self.steps + (step,), self.product * step.event.score, step.event.time)

    def wait(self, until: float) -> "Partial":
        """Шаг-тишина выполнен: окно кончилось в `until`, следующий шаг ждут от этого момента."""
        return Partial(self.steps + (Quiet(until),), self.product, until)


def dominates(a: Partial, b: Partial) -> bool:
    """`a` вытесняет `b` из вершины: последний шаг у `a` не раньше, а произведение не меньше.

    Тогда любое продолжение `b` есть и у `a`, со сроком не раньше и уверенностью не ниже. Если
    равны и последний шаг, и произведение, остаётся начавшееся позже.
    """
    a_last, b_last = a.events[-1].position, b.events[-1].position
    if (a_last, a.product) == (b_last, b.product):
        return a.events[0].position >= b.events[0].position
    return a_last >= b_last and a.product >= b.product


def same_window(a: Partial, b: Partial) -> bool:
    """`a` вытесняет `b` из ожидания тишины, только если окно у них одно и то же.

    Окно, начатое раньше, раньше и кончается, и токен после его конца тишину уже не нарушит.
    Поэтому совпадение с более поздним окном не заменяет совпадение с более ранним.
    """
    return a.time == b.time and dominates(a, b)


def keep(
    partials: list[Partial],
    partial: Partial,
    beats: Callable[[Partial, Partial], bool] = dominates,
) -> None:
    """Добавляет совпадение к несравнимым, если его никто не вытесняет, и убирает вытесненные им."""
    if any(beats(other, partial) for other in partials):
        return
    partials[:] = [other for other in partials if not beats(partial, other)]
    partials.append(partial)


class Engine:
    """Частичные совпадения хранятся между вызовами, поэтому поток можно подавать порциями.

    - Токен, не продолжающий частичное совпадение, пропускается: совпадение ждёт дальше.
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
    - Шаг-тишина ставит совпадение на таймер: оно ждёт `for` секунд от предыдущего шага.
      Запрещённый токен со score не ниже порога тишины, пришедший в окне, снимает ожидание;
      такой же токен не раньше чем за `for` секунд до предыдущего шага тоже. Когда время
      перешло конец окна, шаг выполнен: совпадение входит в вершину за тишиной, и следующий шаг
      ждут от конца окна. Тишина в общую уверенность не входит.
    - Время двигают события и `advance`. `push` сначала выполняет тишины, чьи окна кончились до
      события; `next_deadline` говорит, когда кончится ближайшее окно.
    - Шаблоны сценариев с `order: "any"` идут мимо бора. Их совпадение — набор закрытых шагов:
      токен закрывает любой ещё открытый шаг со своим токеном, если проходит его порог и
      укладывается в задержку от последнего шага. `<дверь> <шаги>` и `<шаги> <дверь>` дают одно
      состояние `{<дверь>, <шаги>}`. Копия, вытеснение и ранний отсев — как в вершине бора;
      совпадение снимается, когда его задержка истекла.
    - Если в один момент завершились несколько совпадений одного сценария, выдаётся не больше
      одного: самое уверенное, а при равной уверенности — начавшееся позже.
    - Срабатывание не выдаётся, если у него есть общий токен с выданным срабатыванием того же
      сценария (токены сравниваются по позиции в потоке) или если с выданного срабатывания
      этого сценария прошло меньше его `cooldown`. Невыданное срабатывание не занимает токены
      и не продлевает cooldown, а совпадение живёт дальше.
    - Время не убывает: событие или `advance` раньше предыдущего времени — ошибка.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._vocabulary = rules.vocabulary
        self._cooldowns = {scenario.name: scenario.cooldown for scenario in rules.scenarios}
        self._min_confidences = {
            scenario.name: rules.confidence_threshold(scenario) for scenario in rules.scenarios
        }
        self._starts = build_graph(rules)
        self._unordered = build_unordered(rules)
        self._active: dict[Node, list[Partial]] = {}  # вершина → несравнимые совпадения в ней
        # (шаблон `any`, номера закрытых шагов) → несравнимые совпадения в этом состоянии
        self._sets: dict[tuple[Unordered, frozenset[int]], list[Partial]] = {}
        # (ребро-тишина, вершина за ним) → совпадения, которые ждут конца её окна
        self._waiting: dict[tuple[Silence, Node], list[Partial]] = {}
        self._silences: set[Silence] = set()  # все рёбра-тишины графа
        nodes = list(self._starts.values())
        while nodes:
            node = nodes.pop()
            self._silences.update(node.silences)
            nodes += [*node.children.values(), *node.silences.values()]
        # тишина → когда пришёл последний запрещённый ею токен: для окна до предыдущего шага
        self._heard: dict[Silence, float] = {}
        self._taken: dict[str, set[int]] = {}  # сценарий → позиции токенов его выданных срабатываний
        self._fired_at: dict[str, float] = {}  # сценарий → время его последнего выданного срабатывания
        self._position = 0
        self._last_time: float | None = None  # None, пока время ни разу не двигали

    @property
    def active_count(self) -> int:
        return sum(
            len(partials)
            for store in (self._active, self._sets, self._waiting)
            for partials in store.values()
        )

    def feed(self, events: Iterable[Event]) -> list[Match]:
        matches: list[Match] = []
        for event in events:
            matches.extend(self.push(event))
        return matches

    def next_deadline(self) -> float | None:
        """Ближайший конец окна тишины; None, если тишины никто не ждёт.

        Шаг выполняется, когда время перейдёт этот срок: событием или `advance`.
        """
        return min(
            (
                partial.time + silence.window
                for (silence, _), partials in self._waiting.items()
                for partial in partials
            ),
            default=None,
        )

    def advance(self, time: float) -> list[Match]:
        """Двигает время без события и выдаёт срабатывания тишин, чьи окна кончились раньше `time`.

        Окно, которое кончается ровно в `time`, ещё открыто: токен с этим же временем может прийти.
        """
        if not math.isfinite(time):
            raise ValueError(f"время {time} не конечное число")
        if self._last_time is not None and time < self._last_time:
            raise ValueError(f"время {time} раньше предыдущего {self._last_time}")
        matches = self._close(time)
        self._last_time = time
        # сроки истекают и без события: такие совпадения снимаются так же, как в `push`
        active = {
            node: [
                partial
                for partial in partials
                if any(time - partial.time <= edge.max_gap for edge in node.children)
            ]
            for node, partials in self._active.items()
        }
        self._active = {node: partials for node, partials in active.items() if partials}
        sets = {
            key: [partial for partial in partials if time - partial.time <= key[0].max_gap]
            for key, partials in self._sets.items()
        }
        self._sets = {key: partials for key, partials in sets.items() if partials}
        self._forget()
        return matches

    def push(self, event: Event) -> list[Match]:
        if event.token not in self._vocabulary:
            raise ValueError(f"токена {event.token!r} нет в словаре")
        if self._last_time is not None and event.time < self._last_time:
            raise ValueError(
                f"время {event.time} токена {event.token!r} раньше предыдущего {self._last_time}"
            )
        matches = self._close(event.time)  # тишины, чьи окна кончились до этого события
        self._last_time = event.time
        step = Step(self._position, event)
        self._position += 1

        # запрещённый токен нарушает тишину. Все ждущие окна идут до этого события или дольше:
        # кончившиеся раньше только что закрыл `_close`
        broken = {
            silence
            for silence in self._silences
            if event.token in silence.absent and event.score >= silence.min_score
        }
        self._waiting = {
            key: partials for key, partials in self._waiting.items() if key[0] not in broken
        }

        active: dict[Node, list[Partial]] = {}
        sets: dict[tuple[Unordered, frozenset[int]], list[Partial]] = {}
        completed: dict[str, list[Partial]] = {}  # сценарий → совпадения, завершённые этим событием

        def collect(pattern: Unordered, filled: frozenset[int], partial: Partial) -> None:
            length = len(pattern.slots)
            if len(filled) == length:
                completed.setdefault(pattern.scenario, []).append(partial)
            elif partial.bound(length) >= pattern.min_confidence:  # ранний отсев
                keep(sets.setdefault((pattern, filled), []), partial)

        for node, partials in self._active.items():
            for partial in partials:
                # срок считается от последнего шага совпадения, а не от последнего токена потока
                elapsed = event.time - partial.time
                if all(elapsed > edge.max_gap for edge in node.children):
                    continue  # сроки всех рёбер из вершины истекли: совпадение снято
                # копия остаётся в вершине и ждёт другого продолжения
                keep(active.setdefault(node, []), partial)
                for edge, child in node.children.items():
                    if (
                        edge.token == event.token
                        and elapsed <= edge.max_gap
                        and event.score >= edge.min_score
                    ):
                        self._enter(active, completed, child, partial.extend(step))
        for start, child in self._starts.items():
            if start.token == event.token and event.score >= start.min_score:
                # новое совпадение задержку не проверяет
                self._enter(active, completed, child, Partial((step,), event.score, event.time))
        for (pattern, filled), partials in self._sets.items():
            for partial in partials:
                if event.time - partial.time > pattern.max_gap:
                    continue  # задержка у открытых шагов одна, и она истекла: совпадение снято
                keep(sets.setdefault((pattern, filled), []), partial)  # копия ждёт другого шага
                for index in pattern.free(filled, event.token, event.score):
                    collect(pattern, filled | {index}, partial.extend(step))
        for pattern in self._unordered:
            # начать набор может любой его шаг
            for index in pattern.free(frozenset(), event.token, event.score):
                collect(pattern, frozenset({index}), Partial((step,), event.score, event.time))
        # токен запоминается после обхода: шаг, который он сам сделал, его тишину не нарушает
        for silence in broken:
            self._heard[silence] = event.time
        # срабатывания выдаются после обхода: из совпадений сценария выбирается самое уверенное
        matches += self._fire(completed, event.time)

        self._active = active
        self._sets = sets
        self._forget()
        return matches

    def _enter(
        self,
        active: dict[Node, list[Partial]],
        completed: dict[str, list[Partial]],
        node: Node,
        partial: Partial,
    ) -> None:
        """Совпадение вошло в вершину: может сработать, ждать рёбер-токенов и окон тишины."""
        if node.scenario is not None:
            completed.setdefault(node.scenario, []).append(partial)
        # ранний отсев; у вершины без рёбер-токенов совпадению в ней ждать нечего
        if node.children and any(
            partial.bound(length) >= threshold for length, threshold in node.ahead
        ):
            keep(active.setdefault(node, []), partial)
        for silence, child in node.silences.items():
            heard = self._heard.get(silence)
            if heard is not None and heard >= partial.time - silence.window:
                continue  # запрещённый токен был незадолго до шага: тишина уже нарушена
            if self._hopeful(child, partial):
                keep(self._waiting.setdefault((silence, child), []), partial, same_window)

    def _hopeful(self, node: Node, partial: Partial) -> bool:
        """Может ли совпадение после тишины сработать в вершине за ней или ниже (ранний отсев)."""
        if node.scenario is not None and partial.confidence >= self._min_confidences[node.scenario]:
            return True
        return any(partial.bound(length) >= threshold for length, threshold in node.ahead)

    def _close(self, now: float) -> list[Match]:
        """Выполняет шаги-тишины, чьи окна кончились раньше `now`, по порядку сроков: за тишиной
        совпадение может сразу ждать следующую."""
        matches: list[Match] = []
        while (deadline := self.next_deadline()) is not None and deadline < now:
            ended: list[tuple[Node, Partial]] = []
            for (silence, child), partials in self._waiting.items():
                ended += [
                    (child, partial)
                    for partial in partials
                    if partial.time + silence.window == deadline
                ]
                partials[:] = [
                    partial for partial in partials if partial.time + silence.window != deadline
                ]
            self._waiting = {key: partials for key, partials in self._waiting.items() if partials}
            completed: dict[str, list[Partial]] = {}
            for child, partial in ended:
                self._enter(self._active, completed, child, partial.wait(deadline))
            matches += self._fire(completed, deadline)
        return matches

    def _fire(self, completed: dict[str, list[Partial]], time: float) -> list[Match]:
        """Срабатывания совпадений, завершённых в момент `time`: не больше одного на сценарий."""
        matches: list[Match] = []
        for scenario, partials in completed.items():
            taken = self._taken.setdefault(scenario, set())
            eligible = [
                partial
                for partial in partials
                if partial.confidence >= self._min_confidences[scenario]
                and taken.isdisjoint(step.position for step in partial.events)
            ]
            if not eligible:
                continue  # уверенность ниже порога или повтор: токен уже вошёл в срабатывание
            fired_at = self._fired_at.get(scenario)
            if fired_at is not None and time - fired_at < self._cooldowns[scenario]:
                continue
            best = max(
                eligible, key=lambda partial: (partial.confidence, partial.events[0].position)
            )
            taken.update(step.position for step in best.events)
            self._fired_at[scenario] = time
            matches.append(Match(scenario, best.steps, best.confidence))
        return matches

    def _forget(self) -> None:
        # новые срабатывания состоят из позиций не раньше начала самого старого активного
        # совпадения, более ранние можно забыть
        oldest = min(
            (
                partial.events[0].position
                for store in (self._active, self._sets, self._waiting)
                for partials in store.values()
                for partial in partials
            ),
            default=self._position,
        )
        for taken in self._taken.values():
            taken.difference_update([position for position in taken if position < oldest])
