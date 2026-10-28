"""Ничья при выборе истории и порядок выдачи одного момента (PLAN.md, Р18)."""

from rule_engine import Engine, Event


def positions(match):
    return [step.position for step in match.steps]


def feed(engine, tokens, score=0.9):
    matches = []
    for index, token in enumerate(tokens):
        matches.append(engine.push(Event(token, score, float(index))))
    return matches


def test_equal_confidence_and_start_prefer_later_steps(make_rules):
    rules = make_rules([{"name": "S", "patterns": [["A", "B", "C"], ["A", "D", "C"]]}])
    for stream in ("ABDC", "ADBC"):
        [match] = feed(Engine(rules), stream)[-1]
        assert positions(match) == [0, 2, 3]


def test_tie_after_zero_score_prefers_later_steps(make_rules):
    # произведения #0 и #1 разные, но после score 0 равны: в вершину «после B» входят оба
    rules = make_rules([{"name": "S", "patterns": [["A", "B", "C"]]}],
                       min_score=0.0, min_confidence=0.0)
    engine = Engine(rules)
    engine.push(Event("A", 0.9, 0.0))
    engine.push(Event("A", 0.5, 1.0))
    engine.push(Event("B", 0.0, 2.0))
    [match] = engine.push(Event("C", 0.9, 3.0))
    assert positions(match) == [1, 2, 3]


def test_one_moment_is_ordered_by_start_then_declaration(make_rules):
    rules = make_rules([
        {"name": "S1", "patterns": [["B", "B", "A"]]},
        {"name": "S2", "patterns": [["B", "A"]]},
    ])
    [match1, match2] = feed(Engine(rules), "BBA")[-1]
    assert (match1.scenario, positions(match1)) == ("S1", [0, 1, 2])
    assert (match2.scenario, positions(match2)) == ("S2", [1, 2])

    rules = make_rules([
        {"name": "S1", "patterns": [["A", "B", "C"]]},
        {"name": "S2", "patterns": [["A", "D", "C"]]},
    ])
    assert [match.scenario for match in feed(Engine(rules), "ABDC")[-1]] == ["S1", "S2"]


def test_added_scenario_is_declared_after_earlier_ones(make_rules):
    engine = Engine(make_rules([{"name": "late", "patterns": [["A", "C"]]}]))
    engine.add_scenario({"name": "early", "patterns": [["A", "B", "C"]]})
    engine.add_scenario({"name": "first", "order": "any", "patterns": [["C", "A"]]})
    matches = feed(engine, "ABC")[-1]
    assert [match.scenario for match in matches] == ["late", "early", "first"]


def test_window_moment_comes_before_the_event(make_rules):
    engine = Engine(make_rules([
        {"name": "S1", "patterns": [["X", "Z"]], "max_gap": 10.0},
        {"name": "S2", "patterns": [["Y", {"absent": ["X"], "for": 2.0}]]},
    ]))
    engine.push(Event("X", 0.9, 0.0))
    engine.push(Event("Y", 0.9, 3.0))
    matches = engine.push(Event("Z", 0.9, 6.0))
    # окно S2 закрылось в 5.0, раньше события; S1 начался раньше, но сработал позже
    assert [(match.scenario, positions(match)) for match in matches] == [("S2", [1]), ("S1", [0, 2])]


def test_windows_with_one_deadline_are_one_moment(make_rules):
    hush = {"absent": ["X"], "for": 2.0}
    engine = Engine(make_rules([
        {"name": "S1", "patterns": [["B", hush]]},
        {"name": "S2", "patterns": [["A", "B", hush]]},
    ]))
    engine.push(Event("A", 0.9, 0.0))
    engine.push(Event("B", 0.9, 1.0))
    matches = engine.advance(4.0)
    assert [(match.scenario, positions(match)) for match in matches] == [("S2", [0, 1]), ("S1", [1])]
