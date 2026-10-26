"""Добавление сценариев между событиями: `uv run python examples/dynamic.py`."""

from pathlib import Path

from rule_engine import Engine, Event, load_rules


def main() -> None:
    engine = Engine(load_rules(Path(__file__).with_name("smart_home.json")))

    def push(token: str, time: float) -> None:
        position = engine.position
        matches = engine.push(Event(token, 0.9, time))
        print(f"#{position} {token} @{time:g} с")
        for match in matches:
            positions = " ".join(f"#{step.position}" for step in match.steps)
            print(f"  ✓ {match.scenario} ({match.confidence:.2f}), шаги: {positions}")

    push("открой", 0.0)
    engine.add_scenario({"name": "дверь: открыть", "patterns": [["открой", "дверь"]]})
    print("Добавлен сценарий «дверь: открыть»; он может продолжить сохранённое «открой».")
    push("шторы", 0.4)  # старое совпадение продолжилось
    push("дверь", 0.8)  # новое правило использует тот же сохранённый префикс
    push("открой", 5.0)
    push("дверь", 5.4)

    engine.add_scenario(
        {
            "name": "движение и шаги",
            "order": "any",
            "patterns": [["<движение>", "<шаги>"]],
        }
    )
    print("Добавлен набор «движение и шаги»; новый токен <движение> добавлен в словарь.")
    push("<движение>", 10.0)
    push("<шаги>", 10.4)

    engine.add_scenario(
        {"name": "вентилятор: включить", "patterns": [["включи", "вентилятор"]]}
    )
    print("Добавлен сценарий «вентилятор: включить»; новый токен «вентилятор» добавлен в словарь.")
    push("включи", 15.0)
    push("вентилятор", 15.4)


if __name__ == "__main__":
    main()
