"""Словарь токенов и сценарии: исходные данные, из которых строится граф."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    tokens: tuple[str, ...]


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
            if not scenario.tokens:
                raise ValueError(f"сценарий {scenario.name!r} не содержит токенов")
            unknown = [token for token in scenario.tokens if token not in self.vocabulary]
            if unknown:
                raise ValueError(f"сценарий {scenario.name!r}: токенов {unknown} нет в словаре")

    @classmethod
    def from_dict(cls, data: dict) -> "RuleSet":
        """`pattern` сценария — токены через пробел."""
        scenarios = tuple(
            Scenario(item["name"], tuple(item["pattern"].split())) for item in data["scenarios"]
        )
        return cls(frozenset(data["vocabulary"]), scenarios, float(data["max_gap"]))


def load_rules(path: Path) -> RuleSet:
    return RuleSet.from_dict(json.loads(path.read_text(encoding="utf-8")))
