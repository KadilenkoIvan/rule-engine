"""Stream format: a line per event or time mark, shared by CLI files and the web form."""

from collections.abc import Iterable, Iterator

from rule_engine.application.views import Frame
from rule_engine.domain.engine import Event


def parse_stream(lines: Iterable[str]) -> Iterator[Event | float]:
    """Parses stream lines: «token score time» or a time mark «time».

    Empty lines and lines starting with # are skipped. The token is everything before the last
    two fields, so it may contain spaces: `<громкий звук> 0.62 12.4`. A mark moves time without
    an event.

    Yields:
        Events and time marks.

    Raises:
        ValueError: a line has another format.
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
    """Checks that a stream line reads the token back unchanged: live sessions are saved as
    streams.

    Raises:
        ValueError: the token is empty, padded with spaces, multi-line or starts with #.
    """
    if token.splitlines() != [token] or token != token.strip() or token.startswith("#"):
        raise ValueError(
            f"токен {token!r} не записать строкой потока: он пустой, с пробелами по краям, с "
            "переводом строки или начинается с #"
        )


def stream_line(frame: Frame) -> str:
    """Frame as a stream line: «token score time» for an event, a time mark for a timer move.

    Numbers are written exactly (`repr`), so `parse_stream` reads the same events and marks.
    """
    if frame.event is None:
        return repr(frame.time)
    event = frame.event
    return f"{event.token} {event.score!r} {event.time!r}"
