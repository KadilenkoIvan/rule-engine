"""Ports: what the use cases expect from an interface."""

from typing import Protocol

from rule_engine.application.views import Frame


class LiveListener(Protocol):
    """Receiver of live session frames."""

    def on_frame(self, frame: Frame) -> None:
        """Frame of a live session: after an event or a timer move.

        Called from the pushing thread or the engine's timer thread, but one at a time and
        strictly in frame order: the next frame waits while it runs.
        """
        ...
