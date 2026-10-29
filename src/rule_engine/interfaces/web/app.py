"""Web interface: builds the controller and hands it to the thin framework layer."""

import argparse
from pathlib import Path

from rule_engine.application.service import RuleEngineService
from rule_engine.interfaces.web.controller import WebController
from rule_engine.interfaces.web.presenter import Presenter
from rule_engine.interfaces.web.server import serve

STATIC = Path(__file__).parent / "static"


class WebInterface:
    """Web server over a folder of example rules and streams."""

    def __init__(self, service: RuleEngineService) -> None:
        self._service = service

    def run(self, argv: list[str]) -> int:
        """Parses the arguments and serves until stopped; returns the exit code.

        Raises:
            ValueError: the examples folder does not exist.
        """
        parser = argparse.ArgumentParser(prog="rule-engine web")
        parser.add_argument("examples", type=Path, help="папка с правилами *.json и потоками *.txt")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        args = parser.parse_args(argv)
        if not args.examples.is_dir():
            raise ValueError(f"папки {str(args.examples)!r} нет")
        controller = WebController(self._service, Presenter(), args.examples)
        print(f"rule-engine: http://{args.host}:{args.port}")
        serve(controller, STATIC, args.host, args.port)
        return 0
