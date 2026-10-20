from pathlib import Path

from rule_engine.cli import main

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_example_stream(capsys):
    assert main(["run", str(EXAMPLES / "smart_home.json"), str(EXAMPLES / "stream.txt")]) == 0
    scenarios = [line.split("  ←  ")[0] for line in capsys.readouterr().out.splitlines()]
    assert scenarios == [
        "свет в ванной: включить",
        "ванна: разогреть",
        "шторы: открыть",
        "утро",
        "чайник: разогреть",
        "свет: выключить",
        "ухожу из дома",
        "дверь: закрыть",
        "активных частичных совпадений в конце потока: 0",
    ]
