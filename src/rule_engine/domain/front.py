"""Фронт — несравнимые совпадения одного места — и очередь их сроков."""

import heapq
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Callable, Iterator
from itertools import count, islice
from typing import Any, Protocol

from rule_engine.domain.matching import Event, Miss, Partial, Place, dominates

# сроки в очереди — суммы с округлением, а снятие проверяется точной формулой, поэтому из очереди
# достаются и сроки чуть позже события; доля от времени события
SLACK = 1e-9


class Front(ABC):
    """Несравнимые совпадения одного места: вершины бора или состояния набора `any`.

    Совпадения идут по возрастанию последнего шага, а произведение score у них строго убывает:
    более старое остаётся, только если оно увереннее (`dominates`). Поэтому новое совпадение
    вытесняет только хвост, сроки истекают с головы, а самое уверенное стоит первым.

    Номера (`front[i]`, `split`, `weakest`) считаются от самого старого живого совпадения.
    Чем места различаются — задержкой, проверкой срока и объяснением снятия, — задают подклассы.
    """

    __slots__ = ("_items", "_head")

    def __init__(self) -> None:
        self._items: list[Partial] = []
        self._head = 0  # совпадения до `_head` сняты по сроку

    @property
    @abstractmethod
    def place(self) -> Place:
        """Место так, как его видят снимок и отказы."""

    @property
    @abstractmethod
    def gap(self) -> float:
        """Задержка места: позже неё от последнего шага совпадению некуда продолжиться."""

    @abstractmethod
    def expired(self, partial: Partial, now: float) -> bool:
        """Истёк ли к моменту `now` срок совпадения этого места."""

    @abstractmethod
    def expiry_misses(self, partial: Partial, event: Event) -> list[Miss]:
        """Чем объяснить снятие совпадения на событии `event`."""

    @abstractmethod
    def release(self) -> None:
        """Убирает опустевший фронт у владельца: пустых фронтов не хранят."""

    def __len__(self) -> int:
        return len(self._items) - self._head

    def __iter__(self) -> Iterator[Partial]:
        return islice(self._items, self._head, None)

    def __getitem__(self, index: int) -> Partial:
        return self._items[self._head + index]

    def deadline(self) -> float:
        """Срок самого старого совпадения: раньше него фронт ничего не теряет."""
        return self[0].last_time + self.gap

    def add(self, partial: Partial) -> int:
        """Добавляет совпадение, кончающееся на текущем событии; возвращает, насколько вырос фронт.

        Вытеснить новое может только последнее: лишь у него последний шаг может быть тем же.
        """
        items, head = self._items, self._head
        if len(items) > head and dominates(items[-1], partial):
            return 0
        size = len(items)
        while len(items) > head and dominates(partial, items[-1]):
            items.pop()
        items.append(partial)
        return len(items) - size

    def pop_first(self) -> Partial:
        partial = self._items[self._head]
        self._head += 1
        if self._head * 2 >= len(self._items):  # сжатие: в среднем O(1) на снятое совпадение
            del self._items[: self._head]
            self._head = 0
        return partial

    def split(self, fits: Callable[[Partial], bool]) -> int:
        """Номер первого совпадения, для которого `fits` истинно; дальше оно истинно у всех."""
        return bisect_left(self._items, True, self._head, key=fits) - self._head

    def weakest(self, start: int, keeps: Callable[[Partial], bool]) -> Iterator[Partial]:
        """Совпадения с хвоста до номера `start`, пока `keeps` ложно.

        Произведения к хвосту убывают, поэтому отсев по произведению начинается с хвоста и
        останавливается на первом совпадении, которое его прошло.
        """
        for index in range(len(self) - 1, start - 1, -1):
            if keeps(self[index]):
                return
            yield self[index]


class Owner(Protocol):
    """Владелец мест одного вида: отдаёт фронт места, при первом входе — новый пустой."""

    def front(self, key: Any) -> Front: ...


class Deadlines:
    """Очередь сроков: у каждого непустого фронта в ней ровно один срок — срок его головы.

    Срок в очереди бывает раньше настоящего: голову вытеснило свежее совпадение или у вершины
    появилось ребро с большей задержкой. Тогда фронт просто вернётся в очередь. Позже настоящего
    срок не бывает, поэтому истёкшее совпадение не пропускается.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Front]] = []
        self._order = count()  # разводит равные сроки, чтобы куча не сравнивала фронты

    def schedule(self, front: Front) -> None:
        heapq.heappush(self._heap, (front.deadline(), next(self._order), front))

    def due(self, now: float) -> list[Front]:
        """Достаёт фронты, у которых к моменту `now` мог истечь срок; проверяет их вызывающий."""
        due = []
        while self._heap and self._heap[0][0] <= now + SLACK * (1 + abs(now)):
            due.append(heapq.heappop(self._heap)[2])
        return due
