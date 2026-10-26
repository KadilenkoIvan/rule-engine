"""Проход потока токенов по графу сценариев и по шаблонам `any`."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from rule_engine.domain.graph import Node, Start, build_graph
from rule_engine.domain.rules import RuleSet
from rule_engine.domain.unordered import Unordered, build_unordered


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


class Partial(NamedTuple):
    """Частичное совпадение: его шаги и произведение их score."""

    steps: tuple[Step, ...]
    product: float

    @property
    def confidence(self) -> float:
        """Среднее геометрическое score шагов."""
        return self.product ** (1 / len(self.steps))

    def bound(self, length: int) -> float:
        """Уверенность в шаблоне длины `length`, если у оставшихся шагов score 1.0: выше не бывает."""
        return self.product ** (1 / length)

    def extend(self, step: Step) -> "Partial":
        return Partial(self.steps + (step,), self.product * step.event.score)


class Waiting(NamedTuple):
    """Частичное совпадение в снимке движка: где оно стоит и до какого момента ждёт."""

    place: Node | tuple[Unordered, frozenset[int]]  # вершина бора или (шаблон `any`, закрытые шаги)
    partial: Partial
    until: float  # секунды: срок продолжения


def dominates(a: Partial, b: Partial) -> bool:
    """`a` вытесняет `b` из вершины: последний шаг у `a` не раньше, а произведение не меньше.

    Тогда любое продолжение `b` есть и у `a`, со сроком не раньше и уверенностью не ниже. Если
    равны и последний шаг, и произведение, остаётся начавшееся позже.
    """
    if (a.steps[-1].position, a.product) == (b.steps[-1].position, b.product):
        return a.steps[0].position >= b.steps[0].position
    return a.steps[-1].position >= b.steps[-1].position and a.product >= b.product


def keep(partials: list[Partial], partial: Partial) -> None:
    """Добавляет совпадение к несравнимым, если его никто не вытесняет, и убирает вытесненные им."""
    if any(dominates(other, partial) for other in partials):
        return
    partials[:] = [other for other in partials if not dominates(partial, other)]
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
        self._taken: dict[str, set[int]] = {}  # сценарий → позиции токенов его выданных срабатываний
        self._fired_at: dict[str, float] = {}  # сценарий → время его последнего выданного срабатывания
        self._position = 0
        self._last_time: float | None = None  # None, пока не пришло ни одного события

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

    def snapshot(self) -> list[Waiting]:
        """Все частичные совпадения: в вершинах бора и в наборах `any`."""
        # в `_active` стоят только вершины с рёбрами: в остальных ждать нечего
        waiting = [
            Waiting(
                node,
                partial,
                partial.steps[-1].event.time + max(edge.max_gap for edge in node.children),
            )
            for node, partials in self._active.items()
            for partial in partials
        ]
        waiting += [
            Waiting(key, partial, partial.steps[-1].event.time + key[0].max_gap)
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
        if event.token not in self._vocabulary:
            raise ValueError(f"токена {event.token!r} нет в словаре")
        if self._last_time is not None and event.time < self._last_time:
            raise ValueError(
                f"время {event.time} токена {event.token!r} раньше предыдущего {self._last_time}"
            )
        self._last_time = event.time
        step = Step(self._position, event)
        self._position += 1

        matches: list[Match] = []
        active: dict[Node, list[Partial]] = {}
        sets: dict[tuple[Unordered, frozenset[int]], list[Partial]] = {}
        completed: dict[str, list[Partial]] = {}  # сценарий → совпадения, завершённые этим событием

        def enter(node: Node, partial: Partial) -> None:
            if node.scenario is not None:
                completed.setdefault(node.scenario, []).append(partial)
            # ранний отсев; у вершины без продолжений `ahead` пуст, и совпадение в ней не остаётся
            if any(partial.bound(length) >= threshold for length, threshold in node.ahead):
                keep(active.setdefault(node, []), partial)

        def collect(pattern: Unordered, filled: frozenset[int], partial: Partial) -> None:
            length = len(pattern.slots)
            if len(filled) == length:
                completed.setdefault(pattern.scenario, []).append(partial)
            elif partial.bound(length) >= pattern.min_confidence:  # ранний отсев
                keep(sets.setdefault((pattern, filled), []), partial)

        def fire(scenario: str, partials: list[Partial]) -> None:
            taken = self._taken.setdefault(scenario, set())
            eligible = [
                partial
                for partial in partials
                if partial.confidence >= self._min_confidences[scenario]
                and taken.isdisjoint(position for position, _ in partial.steps)
            ]
            if not eligible:
                return  # уверенность ниже порога или повтор: токен уже вошёл в выданное срабатывание
            fired_at = self._fired_at.get(scenario)
            if fired_at is not None and event.time - fired_at < self._cooldowns[scenario]:
                return
            best = max(eligible, key=lambda partial: (partial.confidence, partial.steps[0].position))
            taken.update(position for position, _ in best.steps)
            self._fired_at[scenario] = event.time
            matches.append(Match(scenario, best.steps, best.confidence))

        for node, partials in self._active.items():
            for partial in partials:
                # срок считается от последнего шага совпадения, а не от последнего токена потока
                elapsed = event.time - partial.steps[-1].event.time
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
                        enter(child, partial.extend(step))
        for start, child in self._starts.items():
            if start.token == event.token and event.score >= start.min_score:
                enter(child, Partial((step,), event.score))  # новое совпадение задержку не проверяет
        for (pattern, filled), partials in self._sets.items():
            for partial in partials:
                if event.time - partial.steps[-1].event.time > pattern.max_gap:
                    continue  # задержка у открытых шагов одна, и она истекла: совпадение снято
                keep(sets.setdefault((pattern, filled), []), partial)  # копия ждёт другого шага
                for index in pattern.free(filled, event.token, event.score):
                    collect(pattern, filled | {index}, partial.extend(step))
        for pattern in self._unordered:
            # начать набор может любой его шаг
            for index in pattern.free(frozenset(), event.token, event.score):
                collect(pattern, frozenset({index}), Partial((step,), event.score))
        # срабатывания выдаются после обхода: из совпадений сценария выбирается самое уверенное
        for scenario, partials in completed.items():
            fire(scenario, partials)

        self._active = active
        self._sets = sets
        # новые срабатывания состоят из позиций не раньше начала самого старого активного
        # совпадения, более ранние можно забыть
        oldest = min(
            (
                partial.steps[0].position
                for partials in (*active.values(), *sets.values())
                for partial in partials
            ),
            default=self._position,
        )
        for taken in self._taken.values():
            taken.difference_update([position for position in taken if position < oldest])
        return matches
