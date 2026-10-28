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

    def groups(self) -> dict[str, list[tuple[float, int]]]:
        """Токен → (порог, маска номеров) для групп одинаковых шагов.

        Каждый шаг забирает своё событие, поэтому `[<шаги>, <шаги>]` ждёт два разных `<шаги>`.
        Из одинаковых шагов закрывается первый незакрытый: иначе одно и то же совпадение стояло
        бы в двух состояниях, {первый <шаги>} и {второй <шаги>}. Поэтому группе хватает маски:
        шаг, который закроет событие, — младший бит `маска & ~закрытые`.
        """
        masks: dict[Slot, int] = {}
        for index, slot in enumerate(self.slots):
            masks[slot] = masks.get(slot, 0) | 1 << index
        groups: dict[str, list[tuple[float, int]]] = {}
        for slot, mask in masks.items():
            groups.setdefault(slot.token, []).append((slot.min_score, mask))
        return groups


Owners = dict[tuple[tuple[Slot, ...], float], str]  # набор (без порядка) и задержка → сценарий


def build_unordered(rules: RuleSet) -> list[Unordered]:
    """Шаблоны сценариев с `order: "any"`. Один набор у двух сценариев — ошибка, как в боре."""
    patterns: list[Unordered] = []
    insert_unordered(patterns, {}, rules)
    return patterns


def insert_unordered(patterns: list[Unordered], owners: Owners, rules: RuleSet) -> None:
    """Дописывает наборы сценариев `any` из правил; при ошибке ни список, ни `owners` не меняются.

    Набор проверяется и против прежних, и против уже собранных из тех же правил.
    """
    additions: list[Unordered] = []
    pending: Owners = {}
    for scenario in rules.scenarios:
        if scenario.order != "any":
            continue
        for pattern in scenario.patterns:
            slots = tuple(Slot(step.token, rules.threshold(scenario, step)) for step in pattern)
            # у шагов `any` своей задержки нет, поэтому у всех она сценария или корня
            max_gap = rules.gap(scenario, pattern[0])
            key = (tuple(sorted(slots)), max_gap)
            owner = owners.get(key, pending.get(key))
            if owner is not None:
                raise ValueError(
                    f"сценарии {owner!r} и {scenario.name!r} задают один и тот же набор токенов "
                    f"{[slot.token for slot in slots]} в любом порядке с теми же задержкой и "
                    "порогами"
                )
            pending[key] = scenario.name
            additions.append(
                Unordered(scenario.name, slots, max_gap, rules.confidence_threshold(scenario))
            )
    patterns.extend(additions)
    owners.update(pending)
