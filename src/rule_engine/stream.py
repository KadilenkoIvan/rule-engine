"""Распознавание сценария в потоке команд: «по потоку команд приходим к сценарию».

Каждая выполненная команда — событие ``(действие, устройство, тип, комната)``.
Для каждого сценария поддерживается столбец DP локального выравнивания
Смита–Уотермана потока событий с шагами сценария, обновляемый онлайн за
O(число шагов) на событие:

    H[j] = max(0,
               H_prev[j-1] + s(e, T_j),     # событие e — это шаг j
               H_prev[j]   - gap_event,     # лишнее событие (или повтор шага j)
               H[j-1]      - gap_step)      # шаг j пропущен

В ячейке хранится и битовая маска покрытых шагов — по ней видно, какие шаги
остались. Если покрыто достаточно шагов, движок предлагает выполнить
сценарий целиком. Для сценариев с ``ordered=false`` вместо выравнивания
используется покрытие шагов в скользящем окне.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rule_engine.rules import RuleSet
from rule_engine.spec import ScenarioSpec, ScenarioStep, Spec

MATCH = 2.0
PARTIAL = 0.5
MISMATCH = -1.0
GAP_EVENT = 0.6
GAP_STEP = 0.8


@dataclass(frozen=True, slots=True)
class Event:
    ts: float
    action: str
    device: str
    type: str
    room: str | None


@dataclass
class Suggestion:
    scenario: str
    title: str
    matched: list[int]
    remaining: list[int]
    score: float
    completed: bool = False

    def message(self, spec: Spec) -> str:
        steps = spec.scenarios[self.scenario].steps
        done = len(self.matched)
        total = done + len(self.remaining)
        if self.completed:
            return f"Похоже, вы вручную выполнили сценарий «{self.title}» ({total} из {total} шагов)."
        rest = ", ".join(describe_step(steps[i], spec) for i in self.remaining)
        return (
            f"Похоже на сценарий «{self.title}» (выполнено {done} из {total}). "
            f"Осталось: {rest}. Выполнить остальное?"
        )


def describe_step(step: ScenarioStep, spec: Spec) -> str:
    action = spec.actions[step.action].title
    sel = step.target
    if sel.device:
        target = spec.devices[sel.device].name
    else:
        target = spec.device_types[sel.type or ""].title
        if sel.room:
            target += f" ({spec.rooms[sel.room].aliases[0].alias})"
    return f"{action} {target}"


@dataclass
class _Cell:
    score: float = 0.0
    mask: int = 0


@dataclass
class _Track:
    scenario: ScenarioSpec
    column: list[_Cell]
    window: deque[Event] = field(default_factory=deque)
    suggested_mask: int = 0
    last: tuple[float, int] = (0.0, 0)  # (скор, маска) после последнего события


def selector_matches(event: Event, step: ScenarioStep) -> float:
    """Насколько цель события подходит под цель шага: MATCH, PARTIAL или MISMATCH."""
    sel = step.target
    if sel.device:
        return MATCH if sel.device == event.device else MISMATCH
    if sel.type != event.type:
        return MISMATCH
    if sel.room is None or sel.room == event.room:
        return MATCH
    return PARTIAL


def step_similarity(event: Event, step: ScenarioStep) -> float:
    if event.action != step.action:
        return MISMATCH
    return selector_matches(event, step)


class ScenarioTracker:
    def __init__(
        self,
        ruleset: RuleSet,
        min_fraction: float = 0.5,
        min_steps: int = 2,
        session_gap_ms: float = 30 * 60 * 1000,
        window: int = 12,
    ) -> None:
        self.ruleset = ruleset
        self.min_fraction = min_fraction
        self.min_steps = min_steps
        self.session_gap_ms = session_gap_ms
        self.window = window
        self.last_ts: float | None = None
        self.tracks: dict[str, _Track] = {}
        self.antonyms = {a.id: a.antonym for a in ruleset.spec.actions.values() if a.antonym}
        self.refresh()

    def contradicts(self, event: Event, step: ScenarioStep) -> bool:
        """«Включил свет» после «выключил свет» отменяет засчитанный шаг сценария."""
        return self.antonyms.get(step.action) == event.action and selector_matches(event, step) > 0

    def refresh(self) -> None:
        """Пересобрать треки (например, после добавления выученного сценария)."""
        for sid, scenario in self.ruleset.spec.scenarios.items():
            if sid not in self.tracks or self.tracks[sid].scenario is not scenario:
                self.tracks[sid] = _Track(scenario, [_Cell() for _ in range(len(scenario.steps) + 1)])

    def reset(self, scenario_id: str | None = None) -> None:
        for sid, track in self.tracks.items():
            if scenario_id is None or sid == scenario_id:
                track.column = [_Cell() for _ in track.column]
                track.window.clear()
                track.suggested_mask = 0
                track.last = (0.0, 0)

    def observe(self, event: Event) -> list[Suggestion]:
        return self.observe_batch([event])

    def observe_batch(self, events: list[Event]) -> list[Suggestion]:
        """События одного запроса обрабатываются целиком, подсказка — по итогу.

        Иначе «выключи весь свет» могло бы дать подсказку посреди пакета, до
        того как пришло событие, противоречащее уже засчитанному шагу.
        """
        for event in events:
            if self.last_ts is not None and event.ts - self.last_ts > self.session_gap_ms:
                self.reset()
            self.last_ts = event.ts
            for track in self.tracks.values():
                if track.scenario.steps:
                    track.last = (self._align if track.scenario.ordered else self._cover)(track, event)
        if not events:
            return []
        suggestions = []
        for sid, track in self.tracks.items():
            steps = track.scenario.steps
            if not steps:
                continue
            score, mask = track.last
            matched = [i for i in range(len(steps)) if mask >> i & 1]
            if len(matched) < self.min_steps or len(matched) / len(steps) < self.min_fraction:
                continue
            if mask == track.suggested_mask:
                continue  # уже предлагали ровно это
            track.suggested_mask = mask
            remaining = [i for i in range(len(steps)) if not mask >> i & 1]
            suggestions.append(
                Suggestion(sid, track.scenario.title, matched, remaining, round(score, 3), not remaining)
            )
        suggestions.sort(key=lambda s: (-len(s.matched) / (len(s.matched) + len(s.remaining)), -s.score))
        return suggestions

    def _align(self, track: _Track, event: Event) -> tuple[float, int]:
        steps = track.scenario.steps
        prev = track.column
        col = [_Cell()]
        for j in range(1, len(steps) + 1):
            s = step_similarity(event, steps[j - 1])
            options = [
                _Cell(0.0, 0),
                _Cell(prev[j - 1].score + s, prev[j - 1].mask | (1 << (j - 1)) if s > 0 else prev[j - 1].mask),
                _Cell(prev[j].score - GAP_EVENT, prev[j].mask),
                _Cell(col[j - 1].score - GAP_STEP, col[j - 1].mask),
            ]
            if s > 0:  # повтор шага «выключить весь свет» лампой за лампой — не штраф
                options.append(_Cell(prev[j].score, prev[j].mask | (1 << (j - 1))))
            best = max(options, key=lambda c: (c.score, c.mask.bit_count()))
            col.append(best if best.score > 0 else _Cell())
        # Противоречащее событие обрывает выравнивания, где этот шаг уже засчитан.
        for j, step in enumerate(steps):
            if self.contradicts(event, step):
                col = [cell if not cell.mask >> j & 1 else _Cell() for cell in col]
        track.column = col
        best = max(col, key=lambda c: (c.score, c.mask.bit_count()))
        return best.score, best.mask

    def _cover(self, track: _Track, event: Event) -> tuple[float, int]:
        track.window.append(event)
        while len(track.window) > self.window:
            track.window.popleft()
        mask, score = 0, 0.0
        for e in track.window:  # по порядку: более позднее событие перекрывает раннее
            for i, step in enumerate(track.scenario.steps):
                s = step_similarity(e, step)
                if s > 0 and not mask >> i & 1:
                    mask |= 1 << i
                    score += s
                elif mask >> i & 1 and self.contradicts(e, step):
                    mask &= ~(1 << i)
                    score -= MATCH
        return score, mask
