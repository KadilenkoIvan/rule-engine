"""Шаблоны сценариев с `order: "any"`: шаги собираются в любом порядке, без бора."""

from typing import NamedTuple

from rule_engine.domain.rules import RuleSet


class Slot(NamedTuple):
    """Шаг шаблона `any`: токен и порог. Задержка у шагов общая — у сценария."""

    token: str
    min_score: float  # порог score токена


class Unordered(NamedTuple):
    scenario: str
    slots: tuple[Slot, ...]  # в порядке из JSON; порядок прихода не важен
    max_gap: float  # секунды между соседними по времени шагами
    min_confidence: float  # порог общей уверенности сценария

    def free(self, filled: frozenset[int], token: str, score: float) -> list[int]:
        """Номера шагов, которые событие может закрыть, если шаги `filled` уже закрыты.

        Каждый шаг забирает своё событие, поэтому `[<шаги>, <шаги>]` ждёт два разных `<шаги>`.
        Из одинаковых шагов закрывается первый незакрытый: иначе одно и то же совпадение стояло
        бы в двух состояниях, {первый <шаги>} и {второй <шаги>}.
        """
        return [
            index
            for index, slot in enumerate(self.slots)
            if index not in filled
            and slot.token == token
            and score >= slot.min_score
            and all(earlier in filled for earlier in range(index) if self.slots[earlier] == slot)
        ]


def build_unordered(rules: RuleSet) -> list[Unordered]:
    """Шаблоны сценариев с `order: "any"`. Один набор у двух сценариев — ошибка, как в боре."""
    patterns: list[Unordered] = []
    owners: dict[tuple[tuple[Slot, ...], float], str] = {}  # набор и задержка → сценарий
    for scenario in rules.scenarios:
        if scenario.order != "any":
            continue
        for pattern in scenario.patterns:
            slots = tuple(Slot(step.token, rules.threshold(scenario, step)) for step in pattern)
            # у шагов `any` своей задержки нет, поэтому у всех она сценария или корня
            max_gap = rules.gap(scenario, pattern[0])
            key = (tuple(sorted(slots)), max_gap)
            if key in owners:
                raise ValueError(
                    f"сценарии {owners[key]!r} и {scenario.name!r} задают один и тот же набор "
                    f"токенов {[slot.token for slot in slots]} в любом порядке с теми же задержкой "
                    "и порогами"
                )
            owners[key] = scenario.name
            patterns.append(
                Unordered(scenario.name, slots, max_gap, rules.confidence_threshold(scenario))
            )
    return patterns


def insert_unordered(
    patterns: list[Unordered], owners: dict[tuple[tuple[Slot, ...], float], str], rules: RuleSet
) -> None:
    """Дописывает наборы новых правил; индекс сохраняет прежнюю проверку одинаковых наборов."""
    additions = build_unordered(rules)
    pending = {}
    for pattern in additions:
        key = (tuple(sorted(pattern.slots)), pattern.max_gap)
        if key in owners:
            raise ValueError(
                f"сценарии {owners[key]!r} и {pattern.scenario!r} задают один и тот же набор "
                f"токенов {[slot.token for slot in pattern.slots]} в любом порядке с теми же задержкой "
                "и порогами"
            )
        pending[key] = pattern.scenario
    patterns.extend(additions)
    owners.update(pending)
