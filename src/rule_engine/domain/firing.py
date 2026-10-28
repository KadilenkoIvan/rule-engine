"""Выдача срабатываний: порог общей уверенности, повторы и cooldown (Р4, Р7, Р16)."""

from rule_engine.domain.matching import Match, Miss, Partial, Tick
from rule_engine.domain.rules import RuleSet, Scenario


class Firing:
    """Какие из завершённых событием совпадений выдать.

    От сценария на событие выдаётся не больше одного срабатывания: самое уверенное из тех, что
    прошли порог общей уверенности и не делят токен с выданными срабатываниями сценария. Пока у
    сценария идёт cooldown, он молчит; заглушённое срабатывание токены не занимает.
    """

    def __init__(self, rules: RuleSet) -> None:
        self._cooldowns: dict[str, float] = {}
        self._min_confidences: dict[str, float] = {}
        # сценарий → номер объявления: срабатывания с одним началом выдаются в этом порядке
        self._declared: dict[str, int] = {}
        self._taken: dict[str, set[int]] = {}  # сценарий → позиции токенов его выданных срабатываний
        self.taken_count = 0  # сколько позиций во всех `_taken`
        self._fired_at: dict[str, float] = {}  # сценарий → время его последнего выданного срабатывания
        for scenario in rules.scenarios:
            self.register(rules, scenario)

    def __contains__(self, name: str) -> bool:
        return name in self._declared

    def register(self, rules: RuleSet, scenario: Scenario) -> None:
        self._cooldowns[scenario.name] = scenario.cooldown
        self._min_confidences[scenario.name] = rules.confidence_threshold(scenario)
        self._declared[scenario.name] = len(self._declared)

    def fire(self, tick: Tick) -> list[Match]:
        """Срабатывания события: от начавшегося раньше, при равном начале — по объявлению."""
        matches: list[Match] = []
        for scenario, partials in tick.completed.items():
            match = self._fire(tick, scenario, partials)
            if match is not None:
                matches.append(match)
        # порядок обхода задаёт индекс, а не поток, поэтому выдача упорядочена явно
        matches.sort(key=lambda match: (match.steps[0].position, self._declared[match.scenario]))
        return matches

    def forget_before(self, oldest: int) -> None:
        """Забывает позиции раньше `oldest`: в новые срабатывания они уже не войдут."""
        self.taken_count = 0
        for taken in self._taken.values():
            taken.difference_update([position for position in taken if position < oldest])
            self.taken_count += len(taken)

    def _fire(self, tick: Tick, scenario: str, partials: list[Partial]) -> Match | None:
        threshold = self._min_confidences[scenario]
        taken = self._taken.setdefault(scenario, set())
        eligible = [
            partial
            for partial in partials
            if partial.confidence >= threshold
            and taken.isdisjoint(step.position for step in partial.steps)
        ]
        if not eligible:
            # уверенность ниже порога или повтор: токен уже вошёл в выданное срабатывание
            for partial in partials:
                if partial.confidence < threshold:
                    confidence = partial.confidence
                    tick.misses.append(
                        Miss("confidence", None, partial, scenario, threshold, confidence)
                    )
                else:
                    tick.misses.append(Miss("repeat", None, partial, scenario, 0.0, 0.0))
            return None
        # при равной уверенности — начавшееся позже, при равном начале — с более поздними шагами
        best = max(eligible, key=lambda partial: (partial.confidence, partial.positions()))
        now = tick.event.time
        fired_at = self._fired_at.get(scenario)
        if fired_at is not None and now - fired_at < self._cooldowns[scenario]:
            until = fired_at + self._cooldowns[scenario]
            tick.misses.append(Miss("cooldown", None, best, scenario, until, now))
            return None
        taken.update(best.positions())
        self.taken_count += len(best.steps)  # позиции новые: проверено выше
        self._fired_at[scenario] = now
        return Match(scenario, best.steps, best.confidence)
