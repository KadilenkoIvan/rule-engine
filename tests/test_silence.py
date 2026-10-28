import math

import pytest

from rule_engine import Engine, Event, graph_size
from rule_engine.domain.graph import Silence


def hush(*, duration=2.0, before=0.0, absent=("x",), min_score=0.5):
    return {"absent": list(absent), "for": duration, "before": before, "min_score": min_score}


def test_silence_is_an_unsqueezed_boundary(make_rules):
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", "b", hush(), "c", "d"]]}]))
    assert graph_size(engine.starts) == 3
    [start] = engine.starts.values()
    assert [edge.token for edge in start.segment] == ["a", "b"]
    [silence] = start.silences.values()
    assert len(silence.segment) == 1
    assert isinstance(silence.segment[0], Silence)
    [end] = silence.children.values()
    assert [edge.token for edge in end.segment] == ["c", "d"]


@pytest.mark.parametrize("time, fires", [(2.0, False), (math.nextafter(2.0, math.inf), True)])
def test_window_closes_strictly_after_deadline(make_rules, time, fires):
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", hush()]]}]))
    engine.push(Event("a", 0.9, 0.0))
    assert engine.next_deadline() == 2.0
    matches = engine.advance(time)
    assert bool(matches) is fires
    assert engine.position == 1
    if fires:
        [match] = matches
        assert [step.event.token for step in match.steps] == ["a"]
        assert match.confidence == 0.9
        assert engine.next_deadline() is None
    else:
        assert engine.active_count == 1


@pytest.mark.parametrize("time, fires", [(2.0, False), (math.nextafter(2.0, math.inf), True)])
def test_forbidden_event_on_boundary_breaks_window(make_rules, time, fires):
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", hush()]]}]))
    engine.push(Event("a", 0.9, 0.0))
    assert bool(engine.push(Event("x", 0.9, time))) is fires
    assert [miss.reason for miss in engine.misses] == ([] if fires else ["silence"])
    assert engine.next_deadline() is None


def test_next_gap_starts_at_window_end_without_a_new_history_step(make_rules):
    engine = Engine(make_rules([
        {"name": "s", "patterns": [["a", "b", hush(), "c"]]},
    ]))
    engine.push(Event("a", 0.9, 0.0))
    engine.push(Event("b", 0.8, 1.0))
    [timer] = engine._hushes
    original = timer.payload.partial
    assert engine.advance(4.0) == []
    [partial] = [w.partial for w in engine.snapshot() if w.partial.last_time == 3.0]
    assert partial.tail is original.tail
    assert partial.count == 2
    assert partial.last_time == 3.0
    assert partial.tail.step.event.time == 1.0
    [match] = engine.push(Event("c", 0.7, 6.0))
    assert [step.position for step in match.steps] == [0, 1, 2]
    assert match.confidence == pytest.approx((0.9 * 0.8 * 0.7) ** (1 / 3))


@pytest.mark.parametrize("time, score, breaks", [
    (0.0, 0.9, False), (1.0, 0.9, True), (2.0, 0.9, True), (2.0, 0.4, False),
])
def test_before_boundary_and_score(make_rules, time, score, breaks):
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", hush(before=2.0)]]}]))
    engine.push(Event("x", score, time))
    engine.push(Event("a", 0.9, 3.0))
    assert bool(engine.misses) is breaks
    if breaks:
        [miss] = engine.misses
        assert miss.reason == "silence"
        assert (miss.need, miss.got) == (5.0, time)
    assert bool(engine.advance(6.0)) is not breaks


def test_before_excludes_all_own_steps(make_rules):
    engine = Engine(make_rules([
        {"name": "s", "patterns": [["a", "b", hush(before=4.0, absent=("any",))]]},
    ]))
    engine.feed([Event("a", 0.9, 0.0), Event("b", 0.8, 1.0)])
    assert engine.misses == []
    [match] = engine.advance(4.0)
    assert len(match.steps) == 2


def test_shifted_windows_are_not_dominated(make_rules):
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", hush()]]}]))
    engine.push(Event("a", 0.9, 0.0))
    engine.push(Event("a", 0.9, 1.0))
    assert engine.active_count == 2
    assert sorted(w.until for w in engine.snapshot()) == [2.0, 3.0]
    [match] = engine.push(Event("x", 0.9, 2.5))
    assert match.steps[0].position == 0
    [miss] = engine.misses
    assert miss.reason == "silence"
    assert miss.partial.first_position == 1


def test_same_deadline_completions_share_selection_and_cooldown(make_rules):
    engine = Engine(make_rules([{
        "name": "s", "cooldown": 3.0,
        "patterns": [["a", hush(absent=("x",))], ["a", hush(absent=("y",))]],
    }]))
    engine.push(Event("a", 0.9, 0.0))
    [match] = engine.advance(3.0)
    assert match.scenario == "s"
    engine.push(Event("a", 0.9, 3.0))
    # Cooldown отсчитывается от срока 2, а не от вызова advance(3).
    [match] = engine.advance(6.0)
    assert match.steps[0].event.time == 3.0
    assert engine._fired_at["s"] == 5.0


def test_pruning_counts_tokens_but_not_silence(make_rules):
    engine = Engine(make_rules([
        {"name": "s", "patterns": [["a", hush(), "b"]]},
    ], min_score=0.0, min_confidence=0.7))
    engine.push(Event("a", 0.4, 0.0))
    assert engine.active_count == 0
    [miss] = engine.misses
    assert miss.reason == "pruned"
    assert miss.got == pytest.approx(0.4 ** 0.5)


def test_taken_positions_include_sequences_sets_and_windows(make_rules):
    engine = Engine(make_rules([
        {"name": "short", "patterns": [["a", "b"]]},
        {"name": "long", "patterns": [["a", "c", "d"]]},
        {"name": "window", "patterns": [["x", hush(duration=10.0)]]},
        {"name": "set", "order": "any", "patterns": [["a", "f"]]},
    ]))
    engine.push(Event("a", 0.9, 0.0))
    assert [m.scenario for m in engine.push(Event("b", 0.9, 1.0))] == ["short"]
    engine.push(Event("x", 0.9, 2.0))
    assert engine.push(Event("b", 0.9, 3.0)) == []
    assert "repeat" in [miss.reason for miss in engine.misses]


def test_advance_does_not_expire_token_candidates(make_rules):
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", "b"]]}]))
    engine.push(Event("a", 0.9, 0.0))
    assert engine.advance(10.0) == []
    assert engine.active_count == 1
    engine.push(Event("unknown", 0.9, 10.0))
    assert engine.active_count == 0
    assert engine.misses[0].reason == "expired"


@pytest.mark.parametrize("time", [-1.0, math.inf, math.nan])
def test_invalid_time_is_rejected(make_rules, time):
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", hush()]]}]))
    engine.push(Event("a", 0.9, 0.0))
    with pytest.raises(ValueError):
        engine.advance(time)
