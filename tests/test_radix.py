import pytest

from rule_engine import Cursor, Engine, Event, Start, build_graph, graph_size
from rule_engine.domain.engine import Partial, Step


def test_linear_path_is_one_segment_with_intermediate_cursors(make_rules):
    rules = make_rules([{"name": "chain", "patterns": [["a", "b", "c"]]}])
    starts = build_graph(rules)
    assert graph_size(starts) == 1
    [node] = starts.values()
    assert [edge.token for edge in node.segment] == ["a", "b", "c"]
    engine = Engine(rules)
    assert engine.push(Event("a", 0.9, 0.0)) == []
    [waiting] = engine.snapshot()
    assert isinstance(waiting.place, Cursor)
    assert waiting.place.offset == 1
    assert waiting.place.scenario is None
    assert [(edge.token, target.offset) for edge, target in waiting.place.transitions()] == [("b", 2)]
    assert engine.push(Event("b", 0.8, 1.0)) == []
    [match] = engine.push(Event("c", 0.7, 2.0))
    assert match.scenario == "chain"
    assert [step.position for step in match.steps] == [0, 1, 2]
    assert match.confidence == pytest.approx((0.9 * 0.8 * 0.7) ** (1 / 3))


def test_branches_and_scenario_ends_are_segment_boundaries(make_rules):
    rules = make_rules([
        {"name": "prefix", "patterns": [["a", "b"]]},
        {"name": "left", "patterns": [["a", "b", "c", "d"]]},
        {"name": "right", "patterns": [["a", "b", "e", "f"]]},
    ])
    starts = build_graph(rules)
    assert graph_size(starts) == 3
    [node] = starts.values()
    assert node.scenario == "prefix"
    assert [edge.token for edge in node.segment] == ["a", "b"]
    assert {tuple(edge.token for edge in child.segment) for child in node.children.values()} == {
        ("c", "d"), ("e", "f"),
    }
    engine = Engine(rules)
    matches = engine.feed([Event(token, 0.9, float(i)) for i, token in enumerate("abcdef")])
    assert [match.scenario for match in matches] == ["prefix", "left", "right"]


def test_different_step_settings_do_not_share_an_edge(make_rules):
    rules = make_rules([
        {"name": "fast", "patterns": [["a", {"token": "b", "max_gap": 1.0}, "c"]]},
        {"name": "slow", "patterns": [["a", {"token": "b", "max_gap": 3.0}, "c"]]},
        {"name": "strict", "patterns": [["a", {"token": "b", "min_score": 0.9}, "c"]]},
    ])
    [node] = build_graph(rules).values()
    assert len(node.children) == 3
    engine = Engine(rules)
    engine.push(Event("a", 0.9, 0.0))
    assert engine.push(Event("b", 0.8, 2.0)) == []
    assert {miss.reason for miss in engine.misses} == {"late", "score"}
    assert [match.scenario for match in engine.push(Event("c", 0.9, 3.0))] == ["slow"]


def test_pruning_inside_segment_counts_its_terminal_scenario(make_rules):
    rules = make_rules([{"name": "chain", "patterns": [["a", "b", "c"]]}],
                       min_score=0.0, min_confidence=0.8)
    engine = Engine(rules)
    engine.push(Event("a", 0.1, 0.0))
    assert engine.active_count == 0
    [miss] = engine.misses
    assert miss.reason == "pruned"
    assert miss.place.offset == 1
    assert miss.got == pytest.approx(0.1 ** (1 / 3))
    assert miss.need == 0.8


def test_incomparable_candidates_can_exceed_physical_graph_size(make_rules):
    rules = make_rules([{"name": "chain", "patterns": [["a", "b", "c"]]}])
    engine = Engine(rules)
    engine.push(Event("a", 0.9, 0.0))
    engine.push(Event("a", 0.6, 1.0))
    assert engine.active_count == 2 > graph_size(engine.starts)
    [partials] = engine._active.values()
    assert {partial.first_position for partial in partials} == {0, 1}


def test_history_extension_shares_prefix_and_preserves_events():
    first = Step(10, Event("a", 0.9, 1.0))
    second = Step(12, Event("b", 0.8, 2.0))
    initial = Partial.begin(first)
    extended = initial.extend(second)
    sibling = initial.extend(Step(13, Event("b", 0.7, 3.0)))
    assert extended.tail.previous is sibling.tail.previous is initial.tail
    assert extended.tail.step is second
    assert extended.tail.previous.step is first
    assert extended.steps == (first, second)
    assert initial.steps == (first,)
    assert list(extended.backwards()) == [second, first]
    assert extended.count == 2
    assert extended.first_position == 10
    assert extended.product == pytest.approx(0.72)
    assert extended.last_time == 2.0


def test_history_materialization_is_not_used_for_pruning_or_rejection(make_rules, monkeypatch):
    rules = make_rules([{"name": "chain", "patterns": [["a", "b", "c"]]}])
    engine = Engine(rules)

    def forbidden(_):
        raise AssertionError("внутренняя обработка не должна собирать историю")

    monkeypatch.setattr(Partial, "steps", property(forbidden))
    engine.push(Event("a", 0.9, 0.0))
    engine.push(Event("b", 0.8, 1.0))
    engine.push(Event("c", 0.0, 2.0))
    assert engine.misses[-1].reason == "score"


def test_dynamic_splits_move_cursors_and_preserve_an_open_timer(make_rules):
    silence = {"absent": ["x"], "for": 2.0}
    rules = make_rules([{"name": "base", "patterns": [["a", "b", "c", silence, "d", "e"]]}])
    engine = Engine(rules)
    engine.feed([Event(token, 0.9, float(i)) for i, token in enumerate("abc")])
    [timer] = engine._hushes
    window = timer.payload.window
    tail = timer.payload.partial.tail
    engine.add_scenario({"name": "branches", "patterns": [["a", "x"], ["a", "b", "y"]]})
    assert list(engine._hushes) == [timer]
    assert timer.payload.window is window
    assert timer.payload.partial.tail is tail
    assert {tuple(step.token for step in cursor.node.segment[:cursor.offset])
            for cursor in engine._active} == {("a",), ("b",)}
    assert engine.advance(5.0) == []
    after = [waiting.partial for waiting in engine.snapshot() if waiting.partial.last_time == 4.0]
    [partial] = after
    assert partial.tail is tail
    assert partial.tail.step.event.time == 2.0
    matches = engine.feed([Event("d", 0.9, 5.0), Event("e", 0.9, 6.0)])
    assert [match.scenario for match in matches] == ["base"]


def test_dynamic_silence_does_not_open_retroactively(make_rules):
    rules = make_rules([{"name": "base", "patterns": [["a", "b", "c"]]}])
    engine = Engine(rules)
    engine.push(Event("a", 0.9, 0.0))
    engine.add_scenario({"name": "new", "patterns": [["a", {"absent": ["x"], "for": 1.0}]]})
    assert engine.next_deadline() is None
    engine.push(Event("a", 0.9, 1.0))
    assert engine.next_deadline() == 2.0


def test_failed_insertion_does_not_change_graph_vocabulary_or_state(make_rules):
    rules = make_rules([{"name": "base", "patterns": [["a", "b", "c"]]}])
    engine = Engine(rules)
    engine.push(Event("a", 0.9, 0.0))
    snapshot = engine.snapshot()
    starts = dict(engine.starts)
    before = graph_size(engine.starts), engine.vocabulary, list(engine.misses)
    with pytest.raises(ValueError, match="одну и ту же последовательность"):
        engine.add_scenario({"name": "conflict", "patterns": [["new", "token"], ["a", "b", "c"]]})
    assert engine.starts == starts
    assert engine.snapshot() == snapshot
    assert (graph_size(engine.starts), engine.vocabulary, engine.misses) == before
    assert Start("new", 0.5) not in engine.starts
