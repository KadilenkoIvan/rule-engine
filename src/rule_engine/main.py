"""Composition root: picks the interface and gives it the use cases and the arguments."""

import argparse

from rule_engine.application.service import RuleEngineService
from rule_engine.interfaces.base import Interface

INTERFACES = ("cli", "web")


def choose(name: str, service: RuleEngineService) -> Interface:
    """Interface by name: "cli" or "web"."""
    # импорт ленивый: командной строке не нужно поднимать FastAPI
    if name == "cli":
        from rule_engine.interfaces.cli.app import CliInterface

        return CliInterface(service)
    from rule_engine.interfaces.web.app import WebInterface

    return WebInterface(service)


def main(argv: list[str] | None = None) -> int:
    """Entry point of `rule-engine`; returns the exit code."""
    parser = argparse.ArgumentParser(prog="rule-engine")
    parser.add_argument("interface", choices=INTERFACES, help="cli — командная строка, web — веб")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="аргументы интерфейса")
    args = parser.parse_args(argv)
    return choose(args.interface, RuleEngineService()).run(args.args)
