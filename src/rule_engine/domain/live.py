"""Live mode: an engine with its own clock and a timer thread that moves time to deadlines."""

import threading
import time
from collections.abc import Callable
from typing import NamedTuple

from rule_engine.domain.engine import Engine, Event, Held, Match, Miss, Step, Waiting
from rule_engine.domain.graph import Node, Start
from rule_engine.domain.rules import RuleSet
from rule_engine.domain.unordered import Unordered


class Update(NamedTuple):
    """State after an event or a timer move, taken under the same lock."""

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
    """`Engine` with a clock: it stamps event times, a timer thread calls `advance`.

    The event time is read under the lock, otherwise the timer could move time past it. The
    timer thread sleeps until `next_deadline()`; an event wakes it. `on_update` is called
    outside the lock, from the pushing thread or the timer thread, so updates may arrive out
    of order: `Update.number` restores it. An exception in a timer update stops the thread.

    Args:
        rules: the rules.
        on_update: receives every update, timer moves without matches included.
        clock: seconds; must not go back.
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
        """Root edges of the engine's trie."""
        return self._engine.starts

    @property
    def unordered(self) -> list[Unordered]:
        """`any` patterns of the engine."""
        return self._engine.unordered

    @property
    def vocabulary(self) -> frozenset[str]:
        """Current vocabulary of the engine."""
        return self._engine.vocabulary

    def push_now(self, token: str, score: float) -> Update:
        """Pushes an event at the current clock time.

        Returns:
            The same update `on_update` receives.

        Raises:
            ValueError: the engine is closed.
        """
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
        """Stops the timer thread; open silence windows are not closed."""
        with self._wake:
            self._closed = True
            self._wake.notify()
        self._thread.join()

    def _run(self) -> None:
        """Timer thread: sleeps until the next deadline and moves the engine time past it."""
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
        """Numbered update with the engine state; called under the lock."""
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
