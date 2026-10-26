"""Командная строка: показать граф сценариев и прогнать поток токенов."""

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

from rule_engine.engine import Engine, Event, Match, Quiet, Step
from rule_engine.graph import build_graph, render
from rule_engine.rules import load_rules
from rule_engine.unordered import build_unordered


def parse_stream(lines: Iterable[str]) -> Iterator[Event | float]:
    """Строка потока: событие «токен уверенность время» или метка времени «время». Пустые строки
    и строки с # пропускаются.

    Токеном считается всё до двух последних полей, поэтому в нём бывают пробелы:
    `<громкий звук> 0.62 12.4`. Метка — строка из одного числа: время дошло до него, а событий
    не было. По ней движок выполняет шаги-тишины, чьи окна кончились раньше.
    """
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.rsplit(maxsplit=2)
        if len(parts) == 1:
            try:
                time = float(text)
            except ValueError:
                raise ValueError(
                    f"строка {number}: ожидается «токен уверенность время» или «время», получено "
                    f"{text!r}"
                ) from None
            yield time
            continue
        if len(parts) != 3:
            raise ValueError(
                f"строка {number}: ожидается «токен уверенность время» или «время», получено "
                f"{text!r}"
            )
        token, score, time = parts
        yield Event(token, float(score), float(time))


def format_step(step: Step | Quiet) -> str:
    if isinstance(step, Quiet):
        # конец окна — сумма времени шага и длины окна, а у суммы во float бывает хвост:
        # 0.1 + 0.2 = 0.30000000000000004
        return f"[тишина до {round(step.until, 6)} с]"
    return f"[{step.event.token} {step.event.score:.2f}, {step.event.time} с]"


def format_match(match: Match) -> str:
    """`✓ взлом, уверенность 0.85 — [<дверь> 0.90, 0.0 с] [<громкий звук> 0.80, 9.0 с]`:
    у шага — его токен, уверенность и время, у выдержанной тишины — `[тишина до 3.0 с]`."""
    steps = " ".join(format_step(step) for step in match.steps)
    return f"✓ {match.scenario}, уверенность {match.confidence:.2f} — {steps}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rule-engine")
    commands = parser.add_subparsers(dest="command", required=True)
    graph_command = commands.add_parser("graph", help="показать граф, построенный из сценариев")
    graph_command.add_argument("rules", type=Path)
    run_command = commands.add_parser("run", help="прогнать поток и вывести сработавшие сценарии")
    run_command.add_argument("rules", type=Path)
    run_command.add_argument("stream", type=Path)
    args = parser.parse_args(argv)

    rules = load_rules(args.rules)
    if args.command == "graph":
        print(render(build_graph(rules), build_unordered(rules), rules))
        return 0

    engine = Engine(rules)
    # потока-таймера здесь нет: время двигают метки времени событий и строки-метки
    with args.stream.open(encoding="utf-8") as lines:
        for item in parse_stream(lines):
            matches = engine.push(item) if isinstance(item, Event) else engine.advance(item)
            for match in matches:
                print(format_match(match))
    print(f"незавершённых совпадений в конце потока: {engine.active_count}")
    return 0
