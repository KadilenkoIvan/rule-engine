from __future__ import annotations

import itertools
import random

import pytest

from rule_engine.mining import RuleMiner, contains_with_gap, sessionize
from rule_engine.spec import ScenarioSpec, ScenarioStep, TargetSelector
from rule_engine.stream import Event, ScenarioTracker


def _brute_contains(session, pattern, max_gap):
    for positions in itertools.combinations(range(len(session)), len(pattern)):
        if all(session[p] == x for p, x in zip(positions, pattern)) and all(
            b - a - 1 <= max_gap for a, b in zip(positions, positions[1:])
        ):
            return True
    return False


@pytest.mark.parametrize("seed", range(40))
def test_gap_constrained_subsequence_dp_matches_brute_force(seed):
    rng = random.Random(seed)
    alphabet = [("a", "x"), ("b", "x"), ("c", "x")]
    session = [rng.choice(alphabet) for _ in range(rng.randint(0, 9))]
    pattern = [rng.choice(alphabet) for _ in range(rng.randint(1, 3))]
    gap = rng.randint(0, 2)
    assert contains_with_gap(session, pattern, gap) == _brute_contains(session, pattern, gap)


def test_greedy_would_fail_but_dp_succeeds():
    # жадно берём первое «a» на позиции 0, и тогда «b» на позиции 4 слишком далеко;
    # DP находит вхождение a(3) → b(4)
    session = [("a", "x"), ("z", "x"), ("z", "x"), ("a", "x"), ("b", "x")]
    assert contains_with_gap(session, [("a", "x"), ("b", "x")], 0)


def _sessions(habit, seed=5, with_habit=7, total=10):
    rng = random.Random(seed)
    noise = [("turn_on", "kitchen_light"), ("turn_on", "kitchen_kettle"), ("turn_off", "living_tv")]
    sessions = []
    for i in range(total):
        s = [rng.choice(noise) for _ in range(rng.randint(0, 2))]
        if i < with_habit:
            for item in habit:
                s.append(item)
                if rng.random() < 0.3:
                    s.append(rng.choice(noise))
        sessions.append(s)
    return sessions


def test_miner_finds_planted_habit(spec):
    habit = [("turn_off", "bedroom_ac"), ("open", "bedroom_curtains"), ("start", "vacuum")]
    result = RuleMiner(spec, min_support=0.5, max_gap=1).mine(_sessions(habit))
    found = [tuple((st.action, st.target.device) for st in sc.steps) for sc in result.scenarios]
    assert tuple(habit) in found
    mined = next(sc for sc in result.scenarios if tuple((st.action, st.target.device) for st in sc.steps) == tuple(habit))
    assert mined.source == "mined" and mined.support == 0.7
    assert result.rules and all(r.data["confidence"] >= 0.6 for r in result.rules)


def test_miner_skips_habits_already_covered_by_scenarios(spec):
    # «выключить ТВ → закрыть шторы → включить ночник» уже есть в «Спокойной ночи»
    habit = [("turn_off", "living_tv"), ("close", "living_curtains"), ("turn_on", "bedside_lamp")]
    result = RuleMiner(spec, min_support=0.5, max_gap=1).mine(_sessions(habit))
    found = [tuple((st.action, st.target.device) for st in sc.steps) for sc in result.scenarios]
    assert tuple(habit) not in found
    assert any("Спокойной ночи" in note for note in result.notes)


def test_sessionize_by_time_gap():
    events = [
        {"ts": 0, "action": "turn_on", "device": "a"},
        {"ts": 1000, "action": "turn_off", "device": "a"},
        {"ts": 10_000_000, "action": "turn_on", "device": "b"},
    ]
    assert [len(s) for s in sessionize(events, gap_ms=60_000)] == [2, 1]


def _event(action, device, type_, room, ts=0.0):
    return Event(ts, action, device, type_, room)


def test_tracker_suggests_rest_of_unordered_scenario(ruleset):
    tracker = ScenarioTracker(ruleset)
    tracker.observe(_event("turn_off", "living_tv", "tv", "living"))
    suggestions = tracker.observe(_event("close", "front_door", "door_lock", "hall"))
    assert not [s for s in suggestions if s.scenario == "leave_home"]  # 2 из 7 — мало
    tracker.observe(_event("turn_off", "kitchen_light", "light", "kitchen"))
    suggestions = tracker.observe(_event("turn_off", "living_ac", "ac", "living"))
    leave = [s for s in suggestions if s.scenario == "leave_home"]
    assert leave and set(leave[0].matched) == {0, 1, 2, 4}


def test_tracker_ordered_alignment(ruleset, spec):
    movie = spec.scenarios["movie"]
    assert movie.ordered
    tracker = ScenarioTracker(ruleset)
    tracker.observe(_event("close", "living_curtains", "curtains", "living"))
    tracker.observe(_event("turn_on", "kitchen_kettle", "kettle", "kitchen"))  # постороннее событие
    suggestions = tracker.observe(_event("turn_off", "living_chandelier", "light", "living"))
    movie_s = [s for s in suggestions if s.scenario == "movie"]
    assert movie_s and movie_s[0].matched == [0, 1] and movie_s[0].remaining == [2, 3]


def test_tracker_ignores_wrong_order_for_ordered_scenario(ruleset):
    tracker = ScenarioTracker(ruleset, min_fraction=0.75, min_steps=3)
    for event in [
        _event("turn_on", "living_tv", "tv", "living"),
        _event("turn_off", "living_chandelier", "light", "living"),
        _event("close", "living_curtains", "curtains", "living"),
    ]:
        suggestions = tracker.observe(event)
    assert not [s for s in suggestions if s.scenario == "movie"]


def test_engine_suggests_scenario_from_command_stream(engine):
    for text in ("включи свет на кухне", "включи телевизор", "включи кондиционер в спальне"):
        engine.handle(text)
    responses = [
        engine.handle(text)
        for text in ("выключи свет на кухне", "выключи телевизор", "выключи кондиционер в спальне", "запусти пылесос")
    ]
    assert not any(r.suggestions for r in responses[:3])  # 3 из 7 шагов — ещё рано
    assert any("Ухожу из дома" in s for s in responses[3].suggestions)
    # выполнение самого сценария сбрасывает трекер, повторной подсказки нет
    engine.handle("я ухожу")
    assert not engine.handle("выключи свет в ванной").suggestions


def test_mined_scenario_can_be_added_and_called(engine):
    scenario = ScenarioSpec(
        "evening", (), (ScenarioStep("turn_off", TargetSelector(device="living_tv")),
                        ScenarioStep("turn_on", TargetSelector(device="bedside_lamp"))),
        ordered=True, title="Вечер", source="mined",
    )
    engine.add_scenario(scenario, persist=False)
    engine.add_alias("scenario:evening", "вечерний режим")
    response = engine.handle("вечерний режим")
    assert response.state == "SUCCEEDED"
    assert engine.world.state("bedside_lamp") == "on"


def test_mined_scenarios_persist_in_overlay(tmp_path, spec):
    from rule_engine.engine import Engine

    overlay = tmp_path / "overlay.json"
    habit = [("turn_off", "bedroom_ac"), ("open", "bedroom_curtains"), ("start", "vacuum")]
    first = Engine.from_files(overlay_path=overlay)
    for scenario in RuleMiner(first.spec, min_support=0.5, max_gap=1).mine(_sessions(habit)).scenarios:
        first.add_scenario(scenario)
    second = Engine.from_files(overlay_path=overlay)
    mined = [s for s in second.spec.scenarios.values() if s.source == "mined"]
    assert mined
    alias = mined[0].aliases[0].alias
    assert second.understand(alias).decision.chosen.commands[0].scenario == mined[0].id


def test_contradicting_event_unmatches_step(ruleset):
    tracker = ScenarioTracker(ruleset)
    tracker.observe(_event("turn_off", "living_tv", "tv", "living"))
    tracker.observe(_event("turn_off", "kitchen_light", "light", "kitchen"))
    tracker.observe(_event("turn_on", "kitchen_light", "light", "kitchen"))  # передумал
    suggestions = tracker.observe(_event("turn_on", "bedside_lamp", "light", "bedroom"))
    night = [s for s in suggestions if s.scenario == "good_night"]
    assert not night  # «выключить свет» больше не засчитан: 2 из 6


def test_suggestion_message_is_human_readable(engine):
    from rule_engine.stream import Suggestion

    message = Suggestion("leave_home", "Ухожу из дома", [0, 1], [4, 5], 4.0).message(engine.spec)
    assert "закрыть входная дверь" in message and "включить сигнализация" in message


def test_accepting_tracker_offer_runs_scenario(engine):
    for text in ("включи свет на кухне", "включи телевизор", "включи кондиционер в спальне"):
        engine.handle(text)
    for text in ("выключи свет на кухне", "выключи телевизор", "выключи кондиционер в спальне"):
        engine.handle(text)
    response = engine.handle("запусти пылесос")
    assert response.suggestions and engine.session.offer == "leave_home"
    response = engine.handle("да")
    assert response.state == "SUCCEEDED"
    assert engine.world.state("home_alarm") == "armed"
    assert engine.session.offer is None


def test_offer_expires_after_other_command(engine):
    for text in ("включи свет на кухне", "включи телевизор", "включи кондиционер в спальне",
                 "выключи свет на кухне", "выключи телевизор", "выключи кондиционер в спальне", "запусти пылесос"):
        engine.handle(text)
    assert engine.session.offer == "leave_home"
    engine.handle("громкость телевизора 10")  # другая команда — предложение сгорает
    assert engine.session.offer is None
    engine.handle("да")
    assert engine.world.state("home_alarm") == "disarmed"
