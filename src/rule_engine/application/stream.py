"""Формат потока: строка на событие или метку времени. Общий для файла CLI и веб-формы."""

from collections.abc import Iterable, Iterator

from rule_engine.domain.engine import Event


def parse_stream(lines: Iterable[str]) -> Iterator[Event | float]:
    """Строка потока: «токен уверенность время» или метка «время». Пустые строки и строки с #
    пропускаются.

    Токеном считается всё до двух последних полей, поэтому в нём бывают пробелы:
    `<громкий звук> 0.62 12.4`. Метка — строка из одного числа: время прошло без события, и окна
    тишины, кончившиеся раньше него, закрываются.
    """
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.rsplit(maxsplit=2)
        if len(parts) == 1:
            try:
                yield float(text)
            except ValueError:
                raise ValueError(
                    f"строка {number}: ожидается «токен уверенность время» или метка времени, "
                    f"получено {text!r}"
                ) from None
            continue
        if len(parts) != 3:
            raise ValueError(
                f"строка {number}: ожидается «токен уверенность время», получено {text!r}"
            )
        token, score, time = parts
        yield Event(token, float(score), float(time))
