"""Словарь токенов и сценарии: исходные данные, из которых строится граф."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    patterns: tuple[tuple[str, ...], ...]  # альтернативы: сработал любой шаблон — сработал сценарий
    cooldown: float  # секунды: столько сценарий молчит после выданного срабатывания


@dataclass(frozen=True)
class RuleSet:
    vocabulary: frozenset[str]
    scenarios: tuple[Scenario, ...]
    max_gap: float  # секунды: при паузе дольше начинается новая последовательность команд

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_gap) and self.max_gap > 0):
            raise ValueError(f"max_gap {self.max_gap} должен быть положительным конечным числом")
        names: set[str] = set()
        for scenario in self.scenarios:
            if scenario.name in names:
                raise ValueError(f"сценарий {scenario.name!r} объявлен дважды")
            names.add(scenario.name)
            if not scenario.patterns:
                raise ValueError(f"сценарий {scenario.name!r} не содержит шаблонов")
            patterns: set[tuple[str, ...]] = set()
            for pattern in scenario.patterns:
                if not pattern:
                    raise ValueError(f"сценарий {scenario.name!r} содержит пустой шаблон")
                if pattern in patterns:
                    raise ValueError(f"сценарий {scenario.name!r}: шаблон {list(pattern)} задан дважды")
                patterns.add(pattern)
                unknown = [token for token in pattern if token not in self.vocabulary]
                if unknown:
                    raise ValueError(f"сценарий {scenario.name!r}: токенов {unknown} нет в словаре")
            if not (math.isfinite(scenario.cooldown) and scenario.cooldown >= 0):
                raise ValueError(
                    f"сценарий {scenario.name!r}: cooldown {scenario.cooldown} должен быть "
                    "неотрицательным конечным числом"
                )

    @classmethod
    def from_dict(cls, data: dict) -> "RuleSet":
        """Шаблон — список токенов, а не строка через пробел: в токене бывают пробелы.

        `cooldown` сценария необязателен, по умолчанию 0.
        """
        scenarios = tuple(
            Scenario(
                item["name"],
                tuple(tuple(pattern) for pattern in item["patterns"]),
                float(item.get("cooldown", 0.0)),
            )
            for item in data["scenarios"]
        )
        return cls(frozenset(data["vocabulary"]), scenarios, float(data["max_gap"]))


def load_rules(path: Path) -> RuleSet:
    return RuleSet.from_dict(json.loads(path.read_text(encoding="utf-8")))
