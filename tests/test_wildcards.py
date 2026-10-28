import pytest

from rule_engine import Engine, Event


@pytest.mark.parametrize("wildcard, token, matches", [
    ("any", "known", True), ("any", "unknown", True),
    ("any_vac", "known", True), ("any_vac", "unknown", False),
])
@pytest.mark.parametrize("order", ["sequence", "any"])
def test_wildcard_steps(make_rules, wildcard, token, matches, order):
    engine = Engine(make_rules([
        {"name": "s", "order": order, "patterns": [["a", wildcard, "b"]]},
    ], vocabulary=["a", "b", "known"]))
    result = engine.feed([Event("a", 0.9, 0.0), Event(token, 0.8, 1.0), Event("b", 0.7, 2.0)])
    assert bool(result) is matches
    if matches:
        [match] = result
        assert [step.event.token for step in match.steps] == ["a", token, "b"]


@pytest.mark.parametrize("wildcard, token, breaks", [
    ("any", "known", True), ("any", "unknown", True),
    ("any_vac", "known", True), ("any_vac", "unknown", False),
])
def test_wildcard_silence(make_rules, wildcard, token, breaks):
    engine = Engine(make_rules([
        {"name": "s", "patterns": [["a", {"absent": [wildcard], "for": 2.0}]]},
    ], vocabulary=["a", "known"]))
    engine.push(Event("a", 0.9, 0.0))
    engine.push(Event(token, 0.9, 1.0))
    assert bool(engine.advance(3.0)) is not breaks


def test_any_vocabulary_observes_successful_dynamic_insertion(make_rules):
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", "any_vac", "b"]]}]))
    engine.push(Event("a", 0.9, 0.0))
    engine.push(Event("new", 0.9, 0.5))
    engine.add_scenario({"name": "new-scenario", "patterns": [["new", "c"]]})
    assert engine.vocabulary >= {"new", "c"}
    engine.push(Event("new", 0.9, 1.0))
    [match] = engine.push(Event("b", 0.9, 2.0))
    assert match.scenario == "s"
    assert [step.position for step in match.steps] == [0, 2, 3]


def test_overlapping_slots_branch_and_repeated_slots_need_distinct_events(make_rules):
    engine = Engine(make_rules([
        {"name": "s", "order": "any", "patterns": [["a", "a", "any"]]},
    ]))
    assert engine.push(Event("a", 0.9, 0.0)) == []
    assert {filled for _, filled in engine._sets} == {frozenset({0}), frozenset({2})}
    assert engine.push(Event("unknown", 0.9, 1.0)) == []
    [match] = engine.push(Event("a", 0.9, 2.0))
    assert [step.position for step in match.steps] == [0, 1, 2]
    assert engine.push(Event("a", 0.9, 3.0)) == []


@pytest.mark.parametrize("first, second", [("a", "b"), ("b", "a")])
def test_unordered_history_shares_its_prefix(make_rules, first, second):
    engine = Engine(make_rules([{"name": "s", "order": "any", "patterns": [["a", "b", "c"]]}]))
    engine.push(Event(first, 0.9, 0.0))
    [waiting] = engine.snapshot()
    tail = waiting.partial.tail
    engine.push(Event(second, 0.8, 1.0))
    [extended] = [w.partial for w in engine.snapshot() if w.partial.count == 2]
    assert extended.tail.previous is tail
