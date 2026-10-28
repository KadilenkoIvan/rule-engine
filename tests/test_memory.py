"""Состояние движка не растёт на длинном потоке: позиции срабатываний, очередь сроков, фронты."""

import random

from rule_engine import Engine, Event


def queued(engine):
    """У каждого непустого фронта ровно одна запись в очереди сроков."""
    fronts = len(engine._fronts) + sum(len(collector.states) for collector in engine._collectors)
    return len(engine._deadlines) == fronts


def test_long_stream_keeps_taken_positions_and_queue_bounded(make_rules):
    engine = Engine(make_rules([
        {"name": "pair", "patterns": [["a", "b"]]},
        {"name": "set", "order": "any", "patterns": [["a", "c"]]},
        {"name": "quiet", "patterns": [["b", {"absent": ["a"], "for": 0.6}]]},
    ]))
    rng = random.Random(0)
    fired = 0
    for index in range(20000):
        if index % 7 == 0:
            fired += len(engine.advance(index * 0.25))
        else:
            fired += len(engine.push(Event(rng.choice("abcx"), 0.9, index * 0.25)))
        assert engine._taken_count == sum(map(len, engine._taken.values()))
        assert engine._taken_count < 100
        assert queued(engine)
        assert all(len(front.items) <= 2 * len(front) + 1 for front in engine._fronts.values())
    assert fired > 1000


def test_taken_stays_bounded_when_each_event_fires_several_scenarios(make_rules):
    # три срабатывания на событие: позиций прибавляется больше, чем проходит вызовов
    engine = Engine(make_rules([
        {"name": "token", "patterns": [["a"]]},
        {"name": "any", "patterns": [["any"]]},
        {"name": "word", "patterns": [["any_vac"]]},
    ], vocabulary=["a"]))
    for index in range(5000):
        assert len(engine.push(Event("a", 0.9, float(index)))) == 3
        assert engine._taken_count < 20


def test_taken_is_cleared_once_nothing_is_active(make_rules):
    engine = Engine(make_rules([{"name": "pair", "patterns": [["a", "b"]]}]))
    for index in range(50):
        engine.push(Event("ab"[index % 2], 0.9, float(index)))
    engine.push(Event("x", 0.9, 100.0))  # снимает оставшееся совпадение
    assert engine.active_count == 0
    for index in range(engine._taken_count + 1):
        engine.advance(200.0 + index)
    assert engine._taken == {}
    assert engine._deadlines == []
