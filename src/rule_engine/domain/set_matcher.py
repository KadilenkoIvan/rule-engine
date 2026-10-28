"""Сценарии `any` в проходе: совпадения стоят в состояниях наборов — какие шаги уже закрыты."""

from collections.abc import Iterator

from rule_engine.domain.front import Front
from rule_engine.domain.matching import Event, Miss, Partial, Tick
from rule_engine.domain.rules import RuleSet
from rule_engine.domain.unordered import Owners, Unordered, insert_unordered


class SetFront(Front):
    """Совпадения в состоянии набора. Задержка у шагов набора одна — у шаблона."""

    __slots__ = ("collector", "filled")

    def __init__(self, collector: "Collector", filled: int) -> None:
        super().__init__()
        self.collector = collector
        self.filled = filled  # маска закрытых шагов

    @property
    def place(self) -> tuple[Unordered, frozenset[int]]:
        return self.collector.place(self.filled)

    @property
    def gap(self) -> float:
        return self.collector.pattern.max_gap

    def expired(self, partial: Partial, now: float) -> bool:
        return now > partial.last_time + self.gap

    def expiry_misses(self, partial: Partial, event: Event) -> list[Miss]:
        deadline = partial.last_time + self.gap
        # задержка у открытых шагов одна, и она истекла
        reason = "late" if self.collector.free(self.filled, event.token, 1.0) else "expired"
        return [Miss(reason, self.place, partial, None, deadline, event.time)]

    def release(self) -> None:
        del self.collector.states[self.filled]


class Collector:
    """Шаблон `any` в проходе: группы одинаковых шагов и состояния, где стоят совпадения.

    Состояние — маска закрытых шагов: разные порядки прихода дают одно состояние.
    """

    __slots__ = ("pattern", "groups", "full", "states")

    def __init__(self, pattern: Unordered) -> None:
        self.pattern = pattern
        self.groups = pattern.groups()
        self.full = (1 << len(pattern.slots)) - 1  # маска собранного шаблона
        self.states: dict[int, SetFront] = {}  # маска закрытых шагов → совпадения, без пустых

    def free(self, filled: int, token: str, score: float) -> list[int]:
        """Номера шагов, которые событие может закрыть, если закрыты шаги из маски `filled`."""
        free = []
        for min_score, mask in self.groups.get(token, ()):
            open_ = mask & ~filled
            if open_ and score >= min_score:
                free.append((open_ & -open_).bit_length() - 1)
        return sorted(free)

    def low(self, filled: int, token: str, score: float) -> list[int]:
        """Открытые шаги с этим токеном, порог которых `score` не прошёл."""
        return [
            index
            for index in self.free(filled, token, 1.0)
            if score < self.pattern.slots[index].min_score
        ]

    def place(self, filled: int) -> tuple[Unordered, frozenset[int]]:
        """Состояние так, как его видят снимок и отказы: шаблон и номера закрытых шагов."""
        slots = range(len(self.pattern.slots))
        return self.pattern, frozenset(index for index in slots if filled >> index & 1)

    def front(self, filled: int) -> SetFront:
        front = self.states.get(filled)
        if front is None:
            front = self.states[filled] = SetFront(self, filled)
        return front

    def advance(self, tick: Tick) -> None:
        """Закрывает шаги в состояниях, где ждут совпадения.

        Снятые по сроку уже ушли, а задержка у шагов набора одна, поэтому подходят все.
        """
        event = tick.event
        for filled, front in self.states.items():
            for index in self.free(filled, event.token, event.score):
                self._close(tick, front, filled | 1 << index)
            for index in self.low(filled, event.token, event.score):
                threshold = self.pattern.slots[index].min_score
                place = self.place(filled)
                for partial in front:
                    tick.misses.append(Miss("score", place, partial, None, threshold, event.score))

    def start(self, tick: Tick) -> None:
        """Начинает новые наборы: начать может любой шаг."""
        event = tick.event
        for index in self.free(0, event.token, event.score):
            self._begin(tick, 1 << index, Partial((tick.step,), event.score))
        for index in self.low(0, event.token, event.score):
            threshold = self.pattern.slots[index].min_score
            tick.misses.append(Miss("score", self.place(0), None, None, threshold, event.score))

    def _close(self, tick: Tick, front: SetFront, target: int) -> None:
        """Продолжает совпадения фронта в состояние `target`."""
        pattern, step, score = self.pattern, tick.step, tick.event.score
        length = len(pattern.slots)
        if target == self.full:
            tick.complete(pattern.scenario, (partial.extend(step) for partial in front))
            return
        # как в вершине бора: остаётся самое уверенное, отсеянные — с хвоста
        best = front[0].extend(step)
        if best.bound(length) >= pattern.min_confidence:
            tick.enter(self, target, best)
        for partial in front.weakest(
            0, lambda partial: (partial.product * score) ** (1 / length) >= pattern.min_confidence
        ):
            self._prune(tick, target, partial.extend(step))

    def _begin(self, tick: Tick, filled: int, partial: Partial) -> None:
        """Ставит новое совпадение из одного шага в состояние `filled`."""
        if filled == self.full:
            tick.complete(self.pattern.scenario, [partial])
        elif partial.bound(len(self.pattern.slots)) >= self.pattern.min_confidence:  # ранний отсев
            tick.enter(self, filled, partial)
        else:
            self._prune(tick, filled, partial)

    def _prune(self, tick: Tick, filled: int, partial: Partial) -> None:
        bound = partial.bound(len(self.pattern.slots))
        threshold = self.pattern.min_confidence
        tick.misses.append(Miss("pruned", self.place(filled), partial, None, threshold, bound))


class SetMatcher:
    """Сценарии `any`: их шаблоны, индекс шаблонов по токену и совпадения в состояниях."""

    def __init__(self, rules: RuleSet) -> None:
        self.patterns: list[Unordered] = []
        self._owners: Owners = {}
        self._collectors: list[Collector] = []
        self._index: dict[str, list[Collector]] = {}  # токен → шаблоны `any` с ним
        self.add(rules)

    def add(self, rules: RuleSet) -> None:
        """Дописывает наборы сценариев `any` из правил и индексирует их по токенам."""
        known = len(self.patterns)
        insert_unordered(self.patterns, self._owners, rules)
        for pattern in self.patterns[known:]:
            collector = Collector(pattern)
            self._collectors.append(collector)
            for token in collector.groups:
                self._index.setdefault(token, []).append(collector)

    def advance(self, tick: Tick) -> None:
        """Продолжает и начинает наборы шаблонов, где есть токен события."""
        for collector in self._index.get(tick.event.token, ()):
            collector.advance(tick)
            collector.start(tick)

    def all_fronts(self) -> Iterator[SetFront]:
        for collector in self._collectors:
            yield from collector.states.values()
