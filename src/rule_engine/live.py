"""Живой поток: таймер в отдельном потоке выполняет шаги-тишины по часам."""

import threading
import time
from collections.abc import Callable

from rule_engine.engine import Engine, Event, Match
from rule_engine.rules import RuleSet


class LiveEngine:
    """`Engine` для живого потока: события подаёт `push`, а без событий время двигает таймер.

    Таймер ждёт ближайший срок тишины (`Engine.next_deadline`) по часам — Unix time, той же
    шкале, что у меток событий, — и вызывает `advance(time.time())`. Если `push` сдвинул
    ближайший срок, таймер просыпается раньше. `push` и `advance` идут под одной блокировкой.

    Все срабатывания, и от `push`, и от таймера, отдаются через `on_match` вне блокировки, по
    одному и в том порядке, в каком их выдал движок. Поэтому `on_match` не должен вызывать
    `push` и `close`: он ждал бы сам себя.
    """

    def __init__(self, rules: RuleSet, on_match: Callable[[Match], None]) -> None:
        self._engine = Engine(rules)
        self._on_match = on_match
        self._changed = threading.Condition()  # блокировка движка; на ней же таймер ждёт срок
        self._delivering = threading.Lock()  # её держит тот, кто сейчас отдаёт срабатывания
        self._closed = False
        self._timer = threading.Thread(target=self._run, name="rule-engine-timer", daemon=True)
        self._timer.start()

    def push(self, event: Event) -> None:
        with self._changed:
            matches = self._engine.push(event)
            self._changed.notify()  # событие могло начать новое окно тишины или снять ждущее
            # очередь на выдачу занимается до того, как движок освободится: следующий вызов
            # движка отдаст свои срабатывания только после этих
            self._delivering.acquire()
        self._deliver(matches)

    def close(self) -> None:
        """Останавливает таймер. Ждущие тишины остаются невыполненными."""
        with self._changed:
            self._closed = True
            self._changed.notify()
        self._timer.join()

    def _run(self) -> None:
        while True:
            with self._changed:
                while True:
                    if self._closed:
                        return
                    deadline = self._engine.next_deadline()
                    now = time.time()
                    if deadline is not None and now > deadline:
                        break
                    # окно, которое кончается ровно сейчас, ещё открыто: ждём, пока время перейдёт
                    self._changed.wait(None if deadline is None else deadline - now)
                matches = self._engine.advance(now)
                self._delivering.acquire()
            self._deliver(matches)

    def _deliver(self, matches: list[Match]) -> None:
        try:
            for match in matches:
                self._on_match(match)
        finally:
            self._delivering.release()
