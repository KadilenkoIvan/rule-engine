"""Живой режим: движок со своими часами и таймер-потоком для окон тишины (Р9, Р13 в PLAN.md)."""

import threading
import time
from collections.abc import Callable
from typing import NamedTuple

from rule_engine.domain.engine import Engine, Event, Held, Match, Miss, Step, Waiting
from rule_engine.domain.graph import Node, Start
from rule_engine.domain.rules import RuleSet
from rule_engine.domain.unordered import Unordered


class Update(NamedTuple):
    """Что стало после события или хода таймера. Снимок взят под той же блокировкой."""

    # порядковый номер: `on_update` зовётся вне блокировки, и два обновления из разных потоков
    # могут прийти получателю не по порядку
    number: int
    time: float  # секунды по часам движка
    step: Step | None  # None — ход таймера: кончилось окно тишины или срок выдачи
    matches: tuple[Match, ...]
    waiting: tuple[Waiting, ...]
    held: tuple[Held, ...]  # режим поправок: сработавшие сценарии, чья выдача ждёт срока
    misses: tuple[Miss, ...]


class LiveEngine:
    """`Engine` с часами: время событий ставит сам, окна тишины закрывает таймер-поток.

    - Время события берётся из `clock` под блокировкой. Если брать его до блокировки, таймер
      успеет сдвинуть время вперёд, и событие окажется в прошлом.
    - `clock` по умолчанию — `time.monotonic`: он не идёт назад, а время движка не убывает. Время
      в живом режиме нужно только для сроков, поэтому шкала — своя, не Unix time.
    - Таймер-поток спит до `next_deadline()` (конец окна тишины или срок выдачи в режиме
      поправок) и вызывает `advance`. Сроков нет — спит, пока его не
      разбудят. Событие будит его: оно могло открыть окно с более ранним сроком.
    - `on_update` зовётся вне блокировки: из потока, вызвавшего `push_now`, или из таймер-потока.
      Медленный обработчик не держит движок. Порядок восстанавливается по `Update.number`.
      Исключение в обработчике хода таймера останавливает таймер-поток.
    - Ход таймера отдаётся и без срабатываний: окно закрылось, снимок поменялся.
    """

    def __init__(
        self,
        rules: RuleSet,
        on_update: Callable[[Update], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = Engine(rules)
        self._on_update = on_update
        self._clock = clock
        self._wake = threading.Condition()  # блокировка движка и будильник таймера
        self._closed = False
        self._number = 0
        self._thread = threading.Thread(target=self._run, name="rule-engine-timer", daemon=True)
        self._thread.start()

    @property
    def starts(self) -> dict[Start, Node]:
        return self._engine.starts

    @property
    def unordered(self) -> list[Unordered]:
        return self._engine.unordered

    @property
    def vocabulary(self) -> frozenset[str]:
        return self._engine.vocabulary

    def push_now(self, token: str, score: float) -> Update:
        """Событие «сейчас». Возвращает то же обновление, что получает `on_update`."""
        with self._wake:
            if self._closed:
                raise ValueError("движок закрыт")
            event = Event(token, score, self._clock())
            step = Step(self._engine.position, event)
            update = self._update(event.time, step, self._engine.push(event))
            self._wake.notify()
        self._on_update(update)
        return update

    def close(self) -> None:
        """Останавливает таймер-поток. Открытые окна тишины не закрываются."""
        with self._wake:
            self._closed = True
            self._wake.notify()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._wake:
                while True:
                    if self._closed:
                        return
                    deadline = self._engine.next_deadline()
                    now = self._clock()
                    # окно закрывается, когда время больше срока, а не равно ему
                    if deadline is not None and now > deadline:
                        break
                    # пробуждение раньше срока (событие, ложное) — срок перепроверяется
                    self._wake.wait(None if deadline is None else deadline - now)
                update = self._update(now, None, self._engine.advance(now))
            self._on_update(update)

    def _update(self, time: float, step: Step | None, matches: list[Match]) -> Update:
        self._number += 1
        return Update(
            self._number,
            time,
            step,
            tuple(matches),
            tuple(self._engine.snapshot()),
            tuple(self._engine.held),
            tuple(self._engine.misses),
        )
