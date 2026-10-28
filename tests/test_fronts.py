"""Проход по индексам и фронтам: случаи, где одно лучшее продолжение не заменяет остальные."""

from rule_engine import Engine, Event


def positions(match):
    return [step.position for step in match.steps]


def test_every_history_opens_its_own_window_before_silence(make_rules):
    # после any два несравнимых кандидата: x#0 увереннее, a#1 свежее; окно назад видит оба
    hush = {"absent": ["a"], "for": 2.0, "before": 3.0}
    engine = Engine(make_rules([{"name": "s", "patterns": [["any", "b", hush]]}]))
    engine.push(Event("x", 0.9, 0.0))
    engine.push(Event("a", 0.8, 1.0))
    engine.push(Event("b", 0.9, 2.0))
    # у лучшего (#0 #2) окно назад нарушает чужой a#1; у (#1 #2) a#1 — свой шаг
    assert [miss.reason for miss in engine.misses if miss.reason == "silence"] == ["silence"]
    [match] = engine.advance(5.0)
    assert positions(match) == [1, 2]


def test_window_alone_keeps_taken_positions_through_cleanup(make_rules):
    hush = {"absent": ["y"], "for": 50.0}
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", "b"], ["a", "c", hush]]}]))
    engine.push(Event("a", 0.9, 0.0))
    [match] = engine.push(Event("b", 0.9, 1.0))
    assert positions(match) == [0, 1]
    engine.push(Event("c", 0.9, 2.0))  # (#0 #2) открывает окно до 52.0
    engine.push(Event("x", 0.9, 10.0))  # снимает токеновых кандидатов: остаётся одно окно
    assert engine.active_count == 1
    for index in range(10):  # чистка `_taken` успевает пройти не раз
        engine.advance(11.0 + index)
    assert engine.advance(60.0) == []
    assert [miss.reason for miss in engine.misses] == ["repeat"]


def test_added_edge_extends_the_deadline_of_waiting_candidates(make_rules):
    engine = Engine(make_rules([{"name": "short", "patterns": [["a", {"token": "b", "max_gap": 1.0}]]}]))
    engine.push(Event("a", 0.9, 0.0))
    engine.add_scenario({"name": "long", "patterns": [["a", {"token": "c", "max_gap": 5.0}]]})
    [waiting] = engine.snapshot()
    assert waiting.until == 5.0
    engine.push(Event("x", 0.9, 3.0))  # срок в очереди был 1.0: кандидат возвращается в очередь
    assert engine.active_count == 1
    [match] = engine.push(Event("c", 0.9, 4.0))
    assert (match.scenario, positions(match)) == ("long", [0, 2])


def test_any_vac_sees_tokens_added_while_candidates_wait(make_rules):
    engine = Engine(make_rules([{"name": "s", "patterns": [["a", "any_vac"]]}]))
    engine.push(Event("a", 0.9, 0.0))
    assert engine.push(Event("new", 0.9, 1.0)) == []  # пока не в словаре
    engine.add_scenario({"name": "t", "patterns": [["new", "z"]]})
    [match] = engine.push(Event("new", 0.9, 2.0))
    assert (match.scenario, positions(match)) == ("s", [0, 2])


def test_split_moves_fronts_and_new_branch_reaches_them(make_rules):
    engine = Engine(make_rules([{"name": "long", "patterns": [["a", "b", "c", "d"]]}]))
    engine.push(Event("a", 0.9, 0.0))
    engine.push(Event("b", 0.9, 1.0))
    # расщепление после «a b»: кандидат стоит в конце префикса, ему достаётся новое ребро
    engine.add_scenario({"name": "fork", "patterns": [["a", "b", "x"]]})
    [match] = engine.push(Event("x", 0.9, 2.0))
    assert (match.scenario, positions(match)) == ("fork", [0, 1, 2])
    [match] = engine.feed([Event("c", 0.9, 3.0), Event("d", 0.9, 4.0)])
    assert (match.scenario, positions(match)) == ("long", [0, 1, 3, 4])
