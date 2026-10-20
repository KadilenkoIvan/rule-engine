import random

import pytest

from rule_engine import Engine, Event, RuleSet, build_graph, graph_size

RULES = RuleSet.from_dict(
    {
        "vocabulary": [
            "включи", "выключи", "разогрей", "открой",
            "свет", "чайник", "ванну", "шторы",
            "в", "ванной", "спальне", "пожалуйста", "и",
        ],
        "scenarios": [
            {"name": "свет в ванной", "pattern": "включи свет в ванной"},
            {"name": "свет в спальне", "pattern": "включи свет в спальне"},
            {"name": "чайник", "pattern": "разогрей чайник"},
            {"name": "ванна", "pattern": "разогрей ванну"},
            {"name": "шторы", "pattern": "открой шторы"},
            {"name": "утро", "pattern": "открой шторы разогрей чайник"},
        ],
    }
)


def stream(text: str) -> list[Event]:
    return [Event(token, 0.9) for token in text.split()]


def run(text: str) -> list[str]:
    return [match.scenario for match in Engine(RULES).feed(stream(text))]


def test_scenario_fires_on_its_phrase():
    assert run("разогрей чайник") == ["чайник"]


def test_foreign_tokens_between_steps_are_skipped():
    [match] = Engine(RULES).feed(stream("включи пожалуйста свет в ванной"))
    assert match.scenario == "свет в ванной"
    assert [step.position for step in match.steps] == [0, 2, 3, 4]


def test_rules_with_common_start_fire_on_one_stream():
    assert run("разогрей чайник и разогрей ванну") == ["чайник", "ванна"]


def test_longer_rule_continues_after_its_prefix_fired():
    assert run("открой шторы и разогрей чайник") == ["шторы", "утро", "чайник"]


def test_repeated_start_merges_into_one_match_from_the_latest():
    [match] = Engine(RULES).feed(stream("разогрей разогрей чайник"))
    assert [step.position for step in match.steps] == [1, 2]


def test_completed_match_is_consumed():
    assert run("разогрей чайник чайник") == ["чайник"]


def test_chunk_boundaries_do_not_lose_matches():
    events = stream("включи пожалуйста свет в спальне открой шторы и разогрей разогрей чайник")
    expected = Engine(RULES).feed(events)
    for cut in range(len(events) + 1):
        engine = Engine(RULES)
        assert engine.feed(events[:cut]) + engine.feed(events[cut:]) == expected

    engine = Engine(RULES)
    assert [match for event in events for match in engine.push(event)] == expected


def test_active_matches_never_exceed_graph_size():
    limit = graph_size(build_graph(RULES))
    vocabulary = sorted(RULES.vocabulary)
    rng = random.Random(0)
    engine = Engine(RULES)
    for _ in range(5000):
        engine.push(Event(rng.choice(vocabulary), 0.9))
        assert engine.active_count <= limit


def test_unknown_token_raises():
    with pytest.raises(ValueError, match="нет в словаре"):
        Engine(RULES).push(Event("сделай", 0.9))


def test_score_outside_unit_interval_raises():
    with pytest.raises(ValueError, match="вне"):
        Event("свет", 1.5)
