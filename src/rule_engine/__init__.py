"""Быстрый объяснимый поиск команд и объектов + движок правил на конечных автоматах.

Основные точки входа:

* :class:`rule_engine.engine.Engine` — конвейер «текст → команда → план → исполнение»;
* :class:`rule_engine.rules.RuleCompiler` — спецификация → набор правил;
* :class:`rule_engine.mining.RuleMiner` — поток команд → выученные сценарии;
* :mod:`rule_engine.evaluation` — метрики качества, безопасности и скорости.
"""

from rule_engine.engine import Engine, Response, Session
from rule_engine.rules import RuleCompiler, RuleSet
from rule_engine.spec import Spec

__all__ = ["Engine", "Response", "RuleCompiler", "RuleSet", "Session", "Spec", "main"]


def main() -> None:
    from rule_engine.cli import main as cli_main

    raise SystemExit(cli_main())
