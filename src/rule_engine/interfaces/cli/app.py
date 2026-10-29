"""Command line: show the scenario graph and run a token stream."""

import argparse
from pathlib import Path

from rule_engine.application.service import RuleEngineService
from rule_engine.interfaces.cli.text import format_match, render


class CliInterface:
    """Command line: `graph` prints the graph, `run` runs a stream and prints matches."""

    def __init__(self, service: RuleEngineService) -> None:
        self._service = service

    def run(self, argv: list[str]) -> int:
        """Runs the command of `argv`; returns the exit code."""
        args = _parser().parse_args(argv)
        rules = args.rules.read_text(encoding="utf-8")
        if args.command == "graph":
            print(render(self._service.graph(rules)))
            return 0
        self._trace(rules, args.stream)
        return 0

    def _trace(self, rules: str, stream: Path) -> None:
        """Prints the matches of a stream run and what still waits at its end."""
        waiting = held = 0  # пустой поток: ждать нечему
        with stream.open(encoding="utf-8") as lines:
            for frame in self._service.trace(rules, lines).frames:
                for match in frame.matches:
                    # кадр без события — конец окна тишины или срок выдачи
                    print(format_match(match, frame.time if frame.event is None else None))
                waiting, held = len(frame.waiting), len(frame.held)
        print(f"незавершённых совпадений в конце потока: {waiting}")
        if held:
            print(f"ждут выдачи: {held}")


def _parser() -> argparse.ArgumentParser:
    """Arguments of `rule-engine cli`."""
    parser = argparse.ArgumentParser(prog="rule-engine cli")
    commands = parser.add_subparsers(dest="command", required=True)
    graph_command = commands.add_parser("graph", help="показать граф, построенный из сценариев")
    graph_command.add_argument("rules", type=Path)
    run_command = commands.add_parser(
        "run", help="прогнать поток и вывести сработавшие сценарии"
    )
    run_command.add_argument("rules", type=Path)
    run_command.add_argument("stream", type=Path)
    return parser
