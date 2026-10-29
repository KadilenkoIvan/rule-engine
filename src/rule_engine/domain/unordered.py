"""Patterns of scenarios with `order: "any"`: steps come in any order, outside the trie."""

from typing import NamedTuple

from rule_engine.domain.rules import RuleSet, fits


class Slot(NamedTuple):
    """Step of an `any` pattern: token and threshold; the gap is the scenario's, shared."""

    token: str
    min_score: float  # порог score токена


class Unordered(NamedTuple):
    """Pattern of a scenario with `order: "any"`."""

    scenario: str
    slots: tuple[Slot, ...]  # в порядке из JSON; порядок прихода не важен
    max_gap: float  # секунды между соседними по времени шагами
    min_confidence: float  # порог общей уверенности сценария

    def free(
        self, filled: frozenset[int], token: str, score: float, vocabulary: frozenset[str]
    ) -> list[int]:
        """Indexes of steps the event may close when steps `filled` are closed.

        Each step takes its own event. Of equal steps only the first open one closes, so one
        match never stands in two states; different steps the token fits give different states.
        """
        return [
            index
            for index, slot in enumerate(self.slots)
            if index not in filled
            and fits(slot.token, token, vocabulary)
            and score >= slot.min_score
            and all(earlier in filled for earlier in range(index) if self.slots[earlier] == slot)
        ]

    def groups(self) -> dict[str, list[tuple[float, int]]]:
        """Step token → (threshold, mask of indexes) for each group of equal steps.

        `free` on masks: an event closes the first open step of a group, the lowest bit of
        `mask & ~closed`.
        """
        masks: dict[Slot, int] = {}
        for index, slot in enumerate(self.slots):
            masks[slot] = masks.get(slot, 0) | 1 << index
        groups: dict[str, list[tuple[float, int]]] = {}
        for slot, mask in masks.items():
            groups.setdefault(slot.token, []).append((slot.min_score, mask))
        return groups


def build_unordered(rules: RuleSet) -> list[Unordered]:
    """Patterns of scenarios with `order: "any"`.

    Raises:
        ValueError: two scenarios have the same set with the same gap and thresholds.
    """
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
                raise _duplicate(owners[key], scenario.name, slots)
            owners[key] = scenario.name
            patterns.append(
                Unordered(scenario.name, slots, max_gap, rules.confidence_threshold(scenario))
            )
    return patterns


def insert_unordered(
    patterns: list[Unordered], owners: dict[tuple[tuple[Slot, ...], float], str], rules: RuleSet
) -> None:
    """Appends the sets of new rules; `owners` keeps the check of equal sets.

    Raises:
        ValueError: a new set equals a known one or another new one.
    """
    additions = build_unordered(rules)
    pending = {}
    for pattern in additions:
        key = (tuple(sorted(pattern.slots)), pattern.max_gap)
        if key in owners:
            raise _duplicate(owners[key], pattern.scenario, pattern.slots)
        pending[key] = pattern.scenario
    patterns.extend(additions)
    owners.update(pending)


def _duplicate(owner: str, scenario: str, slots: tuple[Slot, ...]) -> ValueError:
    """Error for a set that two scenarios share."""
    return ValueError(
        f"сценарии {owner!r} и {scenario!r} задают один и тот же набор "
        f"токенов {[slot.token for slot in slots]} в любом порядке с теми же задержкой "
        "и порогами"
    )
