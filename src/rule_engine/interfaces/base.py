"""Interface contract: `main.py` picks the CLI or the web by it."""

from typing import Protocol


class Interface(Protocol):
    """User interface started from the command line."""

    def run(self, argv: list[str]) -> int:
        """Parses its command line arguments and runs to the end; returns the exit code."""
        ...
