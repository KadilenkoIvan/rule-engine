"""Словарь токенов и сценарии: исходные данные, из которых строится граф."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

# допустимые ключи JSON на каждом уровне: другой ключ — ошибка, чтобы опечатка не терялась молча
RULES_KEYS = frozenset({"max_gap", "min_score", "min_confidence", "vocabulary", "scenarios"})
SCENARIO_KEYS = frozenset(
    {"name", "patterns", "order", "cooldown", "max_gap", "min_score", "min_confidence"}
)
ORDERS = ("sequence", "any")  # шаги по порядку шаблона или в любом порядке
STEP_KEYS = frozenset({"token", "max_gap", "min_score"})
SILENCE_KEYS = frozenset({"absent", "for", "min_score"})


@dataclass(frozen=True)
class PatternStep:
    """Шаг шаблона. Не путать со `Step` из engine.py: тот — событие потока внутри совпадения."""

    token: str
    max_gap: float | None  # секунды от предыдущего шага; None — задержка сценария или корня
    min_score: float | None  # порог score токена; None — порог сценария или корня


@dataclass(frozen=True)
class SilenceStep:
    """Шаг-тишина: `window` секунд после предыдущего шага и столько же до него не было ни одного
    токена из `absent` со score не ниже порога."""

    absent: tuple[str, ...]  # запрещённые токены
    window: float  # секунды, ключ "for" в JSON
    min_score: float | None  # порог запрещённых токенов; None — порог сценария или корня


@dataclass(frozen=True)
class Scenario:
    name: str
    # альтернативы: сработал любой шаблон — сработал сценарий
    patterns: tuple[tuple[PatternStep | SilenceStep, ...], ...]
    order: str  # "sequence" — шаги идут по порядку шаблона, "any" — в любом порядке
    cooldown: float  # секунды: столько сценарий молчит после выданного срабатывания
    max_gap: float | None  # секунды между шагами, у которых задержка не задана; None — из корня
    min_score: float | None  # порог шагов, у которых он не задан; None — из корня
    min_confidence: float | None  # порог общей уверенности; None — из корня


@dataclass(frozen=True)
class RuleSet:
    vocabulary: frozenset[str]
    scenarios: tuple[Scenario, ...]
    max_gap: float  # секунды между шагами, если задержку не задали ни шаг, ни сценарий
    min_score: float  # порог score токена, если его не задали ни шаг, ни сценарий
    min_confidence: float  # порог общей уверенности, если сценарий не задал свой

    def __post_init__(self) -> None:
        if not positive_finite(self.max_gap):
            raise ValueError(f"max_gap {self.max_gap} должен быть положительным конечным числом")
        if not unit_interval(self.min_score):
            raise ValueError(f"min_score {self.min_score} должен быть в [0, 1]")
        if not unit_interval(self.min_confidence):
            raise ValueError(f"min_confidence {self.min_confidence} должен быть в [0, 1]")
        names: set[str] = set()
        for scenario in self.scenarios:
            if scenario.name in names:
                raise ValueError(f"сценарий {scenario.name!r} объявлен дважды")
            names.add(scenario.name)
            if scenario.max_gap is not None and not positive_finite(scenario.max_gap):
                raise ValueError(
                    f"сценарий {scenario.name!r}: max_gap {scenario.max_gap} должен быть "
                    "положительным конечным числом"
                )
            if scenario.min_score is not None and not unit_interval(scenario.min_score):
                raise ValueError(
                    f"сценарий {scenario.name!r}: min_score {scenario.min_score} должен быть в [0, 1]"
                )
            if scenario.min_confidence is not None and not unit_interval(scenario.min_confidence):
                raise ValueError(
                    f"сценарий {scenario.name!r}: min_confidence {scenario.min_confidence} должен "
                    "быть в [0, 1]"
                )
            if scenario.order not in ORDERS:
                raise ValueError(
                    f"сценарий {scenario.name!r}: order {scenario.order!r} должен быть "
                    '"sequence" или "any"'
                )
            if not scenario.patterns:
                raise ValueError(f"сценарий {scenario.name!r} не содержит шаблонов")
            patterns: set[tuple[tuple, ...]] = set()
            for pattern in scenario.patterns:
                if not pattern:
                    raise ValueError(f"сценарий {scenario.name!r} содержит пустой шаблон")
                first = pattern[0]
                if isinstance(first, SilenceStep):
                    raise ValueError(
                        f"сценарий {scenario.name!r}: шаг-тишина не может быть первым — её окно "
                        "отсчитывается от предыдущего шага"
                    )
                steps = [step for step in pattern if isinstance(step, PatternStep)]
                silences = [step for step in pattern if isinstance(step, SilenceStep)]
                if scenario.order == "any":
                    if silences:
                        raise ValueError(
                            f"сценарий {scenario.name!r}: шаг-тишина бывает только при order "
                            '"sequence"'
                        )
                    for step in steps:
                        if step.max_gap is not None:
                            raise ValueError(
                                f"сценарий {scenario.name!r}: max_gap у шага {step.token!r} не "
                                'задаётся при order "any" — задержка одна на сценарий и считается '
                                "между соседними по времени шагами"
                            )
                elif first.max_gap is not None:
                    raise ValueError(
                        f"сценарий {scenario.name!r}: max_gap у первого шага {first.token!r} "
                        "не задаётся — новое совпадение задержку не проверяет"
                    )
                for step in steps[1:]:
                    if step.max_gap is not None and not positive_finite(step.max_gap):
                        raise ValueError(
                            f"сценарий {scenario.name!r}: max_gap {step.max_gap} у шага "
                            f"{step.token!r} должен быть положительным конечным числом"
                        )
                for step in steps:
                    if step.min_score is not None and not unit_interval(step.min_score):
                        raise ValueError(
                            f"сценарий {scenario.name!r}: min_score {step.min_score} у шага "
                            f"{step.token!r} должен быть в [0, 1]"
                        )
                for silence in silences:
                    if not silence.absent:
                        raise ValueError(
                            f"сценарий {scenario.name!r}: у шага-тишины пустой список absent"
                        )
                    if len(set(silence.absent)) != len(silence.absent):
                        raise ValueError(
                            f"сценарий {scenario.name!r}: в absent {list(silence.absent)} токен "
                            "повторяется"
                        )
                    if not positive_finite(silence.window):
                        raise ValueError(
                            f"сценарий {scenario.name!r}: for {silence.window} у шага-тишины "
                            "должен быть положительным конечным числом"
                        )
                    if silence.min_score is not None and not unit_interval(silence.min_score):
                        raise ValueError(
                            f"сценарий {scenario.name!r}: min_score {silence.min_score} у "
                            "шага-тишины должен быть в [0, 1]"
                        )
                # шаблоны сравниваются по токенам, действующим задержкам и порогам: шаг без max_gap и
                # шаг с тем же max_gap, что у сценария, — один и тот же шаг
                key = tuple(
                    (step.token, self.gap(scenario, step), self.threshold(scenario, step))
                    if isinstance(step, PatternStep)
                    else (tuple(sorted(step.absent)), step.window, self.threshold(scenario, step))
                    for step in pattern
                )
                if scenario.order == "any":
                    # [<дверь>, <звук>] и [<звук>, <дверь>] — один и тот же шаблон
                    key = tuple(sorted(key))
                if key in patterns:
                    raise ValueError(
                        f"сценарий {scenario.name!r}: шаблон {step_names(pattern)} задан дважды"
                    )
                patterns.add(key)
                tokens = [step.token for step in steps]
                tokens += [token for silence in silences for token in silence.absent]
                unknown = [token for token in tokens if token not in self.vocabulary]
                if unknown:
                    raise ValueError(f"сценарий {scenario.name!r}: токенов {unknown} нет в словаре")
            if not (math.isfinite(scenario.cooldown) and scenario.cooldown >= 0):
                raise ValueError(
                    f"сценарий {scenario.name!r}: cooldown {scenario.cooldown} должен быть "
                    "неотрицательным конечным числом"
                )

    def gap(self, scenario: Scenario, step: PatternStep) -> float:
        """Задержка шага: своя; если не задана — сценария; если нет и там — из корня."""
        if step.max_gap is not None:
            return step.max_gap
        if scenario.max_gap is not None:
            return scenario.max_gap
        return self.max_gap

    def threshold(self, scenario: Scenario, step: PatternStep | SilenceStep) -> float:
        """Порог score шага: свой; если не задан — сценария; если нет и там — из корня.

        У шага-тишины это порог запрещённых токенов: токен ниже него тишину не нарушает.
        """
        if step.min_score is not None:
            return step.min_score
        if scenario.min_score is not None:
            return scenario.min_score
        return self.min_score

    def confidence_threshold(self, scenario: Scenario) -> float:
        """Порог общей уверенности сценария: свой; если не задан — из корня."""
        if scenario.min_confidence is not None:
            return scenario.min_confidence
        return self.min_confidence

    @classmethod
    def from_dict(cls, data: dict) -> "RuleSet":
        """Шаблон — список шагов, а не строка через пробел: в токене бывают пробелы.

        Шаг — строка с токеном, объект `{"token": ..., "max_gap": ..., "min_score": ...}` или
        шаг-тишина `{"absent": [...], "for": ..., "min_score": ...}`.
        `max_gap` и `min_score` у шага и у сценария необязательны, `min_confidence`, `cooldown` и
        `order` сценария — тоже; `cooldown` по умолчанию 0, `order` — "sequence". Неизвестный ключ
        на любом уровне — ошибка.
        """
        check_keys(data, RULES_KEYS, "корень правил")
        scenarios = tuple(parse_scenario(item) for item in data["scenarios"])
        return cls(
            frozenset(data["vocabulary"]),
            scenarios,
            float(data["max_gap"]),
            float(data["min_score"]),
            float(data["min_confidence"]),
        )


def positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0


def unit_interval(value: float) -> bool:
    return 0.0 <= value <= 1.0


def step_names(pattern: tuple[PatternStep | SilenceStep, ...]) -> list[str]:
    """Шаги шаблона для сообщений об ошибках: токен или «тишина N с»."""
    return [
        step.token if isinstance(step, PatternStep) else f"тишина {step.window:g} с"
        for step in pattern
    ]


def check_keys(item: dict, known: frozenset[str], where: str) -> None:
    unknown = sorted(item.keys() - known)
    if unknown:
        raise ValueError(f"{where}: неизвестные ключи {unknown}")


def parse_scenario(item: dict) -> Scenario:
    name = item["name"]
    check_keys(item, SCENARIO_KEYS, f"сценарий {name!r}")
    return Scenario(
        name,
        tuple(tuple(parse_step(name, step) for step in pattern) for pattern in item["patterns"]),
        item.get("order", "sequence"),
        float(item.get("cooldown", 0.0)),
        float(item["max_gap"]) if "max_gap" in item else None,
        float(item["min_score"]) if "min_score" in item else None,
        float(item["min_confidence"]) if "min_confidence" in item else None,
    )


def parse_step(scenario: str, item: str | dict) -> PatternStep | SilenceStep:
    if isinstance(item, str):
        return PatternStep(item, None, None)
    if "absent" in item:
        check_keys(item, SILENCE_KEYS, f"сценарий {scenario!r}, шаг-тишина")
        return SilenceStep(
            tuple(item["absent"]),
            float(item["for"]),
            float(item["min_score"]) if "min_score" in item else None,
        )
    token = item["token"]
    check_keys(item, STEP_KEYS, f"сценарий {scenario!r}, шаг {token!r}")
    return PatternStep(
        token,
        float(item["max_gap"]) if "max_gap" in item else None,
        float(item["min_score"]) if "min_score" in item else None,
    )


def load_rules(path: Path) -> RuleSet:
    return RuleSet.from_dict(json.loads(path.read_text(encoding="utf-8")))
