"""Командная строка: показать граф сценариев и прогнать поток токенов."""

import argparse
from pathlib import Path

from rule_engine.application.service import RuleEngineService
from rule_engine.interfaces.cli.text import format_match, render


class CliInterface:
    def __init__(self, service: RuleEngineService) -> None:
        self._service = service

    def run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser(prog="rule-engine cli")
        commands = parser.add_subparsers(dest="command", required=True)
        graph_command = commands.add_parser("graph", help="показать граф, построенный из сценариев")
        graph_command.add_argument("rules", type=Path)
        run_command = commands.add_parser(
            "run", help="прогнать поток и вывести сработавшие сценарии"
        )
        run_command.add_argument("rules", type=Path)
        run_command.add_argument("stream", type=Path)
        args = parser.parse_args(argv)

        rules = args.rules.read_text(encoding="utf-8")
        if args.command == "graph":
            print(render(self._service.graph(rules)))
            return 0

        waiting = 0  # пустой поток: ждать нечему
        with args.stream.open(encoding="utf-8") as lines:
            for frame in self._service.trace(rules, lines).frames:
                for match in frame.matches:
                    # кадр без события — конец окна тишины
                    print(format_match(match, frame.time if frame.event is None else None))
                waiting = len(frame.waiting)
        print(f"незавершённых совпадений в конце потока: {waiting}")
        return 0
