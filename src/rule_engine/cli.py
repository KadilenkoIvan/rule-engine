"""Командная строка: показать граф сценариев и прогнать поток токенов."""

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

from rule_engine.engine import Engine, Event, Match
from rule_engine.graph import build_graph, render
from rule_engine.rules import load_rules


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


def format_match(match: Match) -> str:
    """`✓ взлом, уверенность 0.85 — [<дверь> 0.90, 0.0 с] [<громкий звук> 0.80, 9.0 с]`:
    у шага — его токен, уверенность и время."""
    steps = " ".join(
        f"[{step.event.token} {step.event.score:.2f}, {step.event.time} с]" for step in match.steps
    )
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
        print(render(build_graph(rules), rules))
        return 0

    engine = Engine(rules)
    with args.stream.open(encoding="utf-8") as lines:
        for event in parse_stream(lines):
            for match in engine.push(event):
                print(format_match(match))
    print(f"незавершённых совпадений в конце потока: {engine.active_count}")
    return 0
