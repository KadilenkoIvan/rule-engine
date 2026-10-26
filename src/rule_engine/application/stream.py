"""Формат потока: строка на событие. Общий для файла CLI и веб-формы."""

from collections.abc import Iterable, Iterator

from rule_engine.domain.engine import Event


def parse_stream(lines: Iterable[str]) -> Iterator[Event]:
    """Строка потока: «токен уверенность время». Пустые строки и строки с # пропускаются.

    Токеном считается всё до двух последних полей, поэтому в нём бывают пробелы:
    `<громкий звук> 0.62 12.4`.
    """
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.rsplit(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(
                f"строка {number}: ожидается «токен уверенность время», получено {text!r}"
            )
        token, score, time = parts
        yield Event(token, float(score), float(time))
