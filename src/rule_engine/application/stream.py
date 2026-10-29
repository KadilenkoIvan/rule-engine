"""Формат потока: строка на событие или метку времени. Общий для файла CLI и веб-формы."""

from collections.abc import Iterable, Iterator

from rule_engine.application.views import Frame
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


def check_token(token: str) -> None:
    """Токен, который строка потока прочитает обратно тем же: живая сессия пишется в поток."""
    if token.splitlines() != [token] or token != token.strip() or token.startswith("#"):
        raise ValueError(
            f"токен {token!r} не записать строкой потока: он пустой, с пробелами по краям, с "
            "переводом строки или начинается с #"
        )


def stream_line(frame: Frame) -> str:
    """Кадр строкой потока: событие — «токен уверенность время», ход таймера — метка времени.

    Числа записываются точно (`repr`), поэтому `parse_stream` даст те же события и метки.
    """
    if frame.event is None:
        return repr(frame.time)
    event = frame.event
    return f"{event.token} {event.score!r} {event.time!r}"
