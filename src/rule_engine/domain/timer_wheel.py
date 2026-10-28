"""Иерархическое колесо таймеров: сроки окон тишины (Р16 в PLAN.md).

Уровень 0 — `size` ячеек по `tick` секунд, ячейка каждого следующего уровня в `size` раз шире.
Уровень создаётся, когда срок не помещается в уже созданные. Номера тиков — целые: у таймера
k = ⌊deadline / tick⌋, у текущего времени c = ⌊now / tick⌋, на уровне L номер ячейки —
k // size^L. Целочисленное деление не даёт уровням разойтись из-за округления float.

Колесо только раскладывает таймеры по ячейкам. Сработал ли таймер, решает точное сравнение
`deadline < now`: ячейка шириной `tick`, а сроки — float.
"""

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

TICK = 0.01  # секунды: ширина ячейки уровня 0
SIZE = 64  # ячеек на уровне


@dataclass(eq=False)
class Timer(Generic[T]):
    """Изменяемый и сравнивается по identity: его отменяют по ссылке."""

    deadline: float  # секунды: таймер срабатывает, когда время станет больше
    payload: T
    order: int  # номер постановки: при равном сроке раньше срабатывает поставленный раньше
    cancelled: bool = False
    fired: bool = False


class TimerWheel(Generic[T]):
    """Вставка и отмена — O(уровней); `advance` разбирает не больше `size` ячеек на уровень."""

    def __init__(self, tick: float = TICK, size: int = SIZE) -> None:
        if not (math.isfinite(tick) and tick > 0):
            raise ValueError(f"tick {tick} должен быть положительным конечным числом")
        if size < 2:
            raise ValueError(f"size {size} должен быть не меньше 2")
        self._width = tick
        self._size = size
        self._levels: list[list[list[Timer[T]]]] = []  # уровень → ячейки → таймеры
        self._now: float | None = None  # None, пока не было `advance`
        self._tick = 0  # c: номер текущего тика уровня 0
        self._count = 0  # ждущие таймеры: не отменённые и не сработавшие
        self._order = 0

    @property
    def now(self) -> float | None:
        return self._now

    def __len__(self) -> int:
        return self._count

    def schedule(self, deadline: float, payload: T) -> Timer[T]:
        if self._now is None:
            raise ValueError("время колеса не задано: сначала вызовите advance")
        if not math.isfinite(deadline):
            raise ValueError(f"срок {deadline} не конечное число")
        if deadline <= self._now:
            raise ValueError(f"срок {deadline} должен быть позже текущего времени {self._now}")
        timer = Timer(deadline, payload, self._order)
        self._order += 1
        self._count += 1
        self._insert(timer)
        return timer

    def cancel(self, timer: Timer[T]) -> None:
        """Ставит флаг; из ячейки таймер выбрасывается, когда до неё дойдёт разбор."""
        if timer.cancelled:
            raise ValueError(f"таймер со сроком {timer.deadline} уже отменён")
        if timer.fired:
            raise ValueError(f"таймер со сроком {timer.deadline} уже сработал")
        timer.cancelled = True
        self._count -= 1

    def advance(self, now: float) -> list[Timer[T]]:
        """Двигает время; возвращает сработавшие таймеры (`deadline < now`) по сроку и постановке.

        Первый вызов только задаёт начальное время.
        """
        if not math.isfinite(now):
            raise ValueError(f"время {now} не конечное число")
        if self._now is None:
            self._now, self._tick = now, self._index(now)
            return []
        if now < self._now:
            raise ValueError(f"время {now} раньше текущего {self._now}")
        old, self._now, self._tick = self._tick, now, self._index(now)
        due: list[Timer[T]] = []
        # сверху вниз: таймеры старших уровней спускаются в ячейки, которые ещё будут разобраны
        for level in reversed(range(len(self._levels))):
            scale = self._size**level
            first, last = old // scale, self._tick // scale
            if level > 0:
                # ячейка текущего тика старшего уровня всегда пуста: её таймеры поместились бы ниже
                first += 1
            self._drain(level, first, last, due)
        due.sort(key=lambda timer: (timer.deadline, timer.order))
        for timer in due:
            timer.fired = True
        self._count -= len(due)
        return due

    def next_deadline(self) -> float | None:
        """Ближайший срок ждущего таймера.

        На каждом уровне берётся первая непустая ячейка, затем наименьший срок по уровням: таймер,
        давно поставленный на старший уровень, может истечь раньше свежего таймера уровня 0.
        """
        best: float | None = None
        for level, slots in enumerate(self._levels):
            scale = self._size**level
            current = self._tick // scale
            first = current if level == 0 else current + 1
            for k in range(first, current + self._size):
                index = k % self._size
                bucket = [timer for timer in slots[index] if not timer.cancelled]
                slots[index] = bucket  # отменённые выбрасываются здесь
                if bucket:
                    deadline = min(timer.deadline for timer in bucket)
                    best = deadline if best is None else min(best, deadline)
                    break
        return best

    def _index(self, time: float) -> int:
        return math.floor(time / self._width)

    def _insert(self, timer: Timer[T]) -> None:
        """На самый нижний уровень, где номер ячейки меньше чем на `size` впереди текущего."""
        k = self._index(timer.deadline)
        level = 0
        while k // self._size**level - self._tick // self._size**level >= self._size:
            level += 1
        while len(self._levels) <= level:
            self._levels.append([[] for _ in range(self._size)])
        self._levels[level][k // self._size**level % self._size].append(timer)

    def _place(self, timer: Timer[T], due: list[Timer[T]]) -> None:
        """Таймер из разобранной ячейки: отменён — выбросить, срок прошёл — сработал, иначе ниже."""
        if timer.cancelled:
            return
        if timer.deadline < self._now:
            due.append(timer)
        else:
            self._insert(timer)

    def _drain(self, level: int, first: int, last: int, due: list[Timer[T]]) -> None:
        """Разбирает ячейки уровня с номерами first..last, но каждую не больше одного раза."""
        slots = self._levels[level]
        for k in range(first, first + min(last - first + 1, self._size)):
            index = k % self._size
            bucket = slots[index]
            if bucket:
                slots[index] = []
                for timer in bucket:
                    self._place(timer, due)
