"""Контракт интерфейса: по нему `main.py` выбирает CLI или веб."""

from typing import Protocol


class Interface(Protocol):
    def run(self, argv: list[str]) -> int:
        """Разбирает свои аргументы командной строки и работает до конца; возвращает код выхода."""
        ...
