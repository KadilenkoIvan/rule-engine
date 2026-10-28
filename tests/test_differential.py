"""Дифференциальная проверка: движок и эталон с полным обходом (tests/reference.py).

Срабатывания сравниваются по порядку. Снимок и отказы — как мультимножества: их порядок не
входит в контракт (PLAN.md, Р18). Места сравниваются по логическому пути в графе, потому что у
движка и эталона графы свои.
"""

import random
from collections import Counter

import pytest
from reference import ReferenceEngine

from rule_engine import Cursor, Engine, Event, RuleSet
from rule_engine.domain.engine import Window

TOKENS = ["a", "b", "c", "d"]
STEP_TOKENS = [*TOKENS, "any", "any_vac"]
ABSENT = ["a", "b", "x", "any", "any_vac"]
STREAM = [*TOKENS, "e", "x", "unknown"]  # e — только в словаре, unknown — вне словаря
SCORES = [0.0, 0.3, 0.5, 0.8, 0.9, 1.0]


def random_step(rng, order):
    token = rng.choice(STEP_TOKENS)
    if rng.random() < 0.6:
        return token
    step = {"token": token}
    if rng.random() < 0.5:
        step["min_score"] = rng.choice([0.0, 0.3, 0.6, 0.9])
    if order == "sequence" and rng.random() < 0.4:
        step["max_gap"] = rng.choice([0.5, 1.0, 2.0, 5.0])
    return step


def random_silence(rng):
    silence = {"absent": rng.sample(ABSENT, rng.randint(1, 2)), "for": rng.choice([0.5, 1.0, 2.0, 4.0])}
    if rng.random() < 0.5:
        silence["before"] = rng.choice([0.5, 1.0, 3.0])
    if rng.random() < 0.3:
        silence["min_score"] = rng.choice([0.0, 0.5, 0.9])
    return silence


def random_pattern(rng, order, extra):
    tokens = [*STEP_TOKENS, *extra]
    pattern = []
    for index in range(rng.randint(1, 4)):
        step = random_step(rng, order)
        if index == 0 and isinstance(step, dict):
            step.pop("max_gap", None)
        if extra and rng.random() < 0.2:
            step = rng.choice(tokens)
        pattern.append(step)
    if order == "sequence":
        for index in range(len(pattern) - 1, 0, -1):
            if rng.random() < 0.3:
                pattern.insert(index, random_silence(rng))
        if rng.random() < 0.2:
            pattern.append(random_silence(rng))
    return pattern


def random_scenario(rng, name, extra=()):
    order = "any" if rng.random() < 0.3 else "sequence"
    scenario = {
        "name": name,
        "order": order,
        "patterns": [random_pattern(rng, order, extra) for _ in range(rng.randint(1, 2))],
    }
    if rng.random() < 0.5:
        scenario["max_gap"] = rng.choice([0.5, 1.0, 3.0, 6.0])
    if rng.random() < 0.5:
        scenario["min_confidence"] = rng.choice([0.0, 0.3, 0.6, 0.85])
    if rng.random() < 0.3:
        scenario["min_score"] = rng.choice([0.0, 0.5])
    if rng.random() < 0.3:
        scenario["cooldown"] = rng.choice([1.0, 3.0])
    return scenario


def random_rules(rng, count=7):
    """Случайные правила без конфликтов: одинаковые шаблоны ловят и проверка правил, и бор."""
    while True:
        scenarios = [random_scenario(rng, f"s{index}") for index in range(rng.randint(1, count))]
        try:
            rules = RuleSet.from_dict({
                "max_gap": 2.0, "min_score": 0.3, "min_confidence": 0.3,
                "vocabulary": [*TOKENS, "e", "x"], "scenarios": scenarios,
            })
            ReferenceEngine(rules)
        except ValueError:
            continue
        return rules


def paths(engine):
    """Курсор → логический путь от корня: рёбра шагов до этой позиции."""
    result = {}

    def walk(node, prefix):
        for offset in range(1, len(node.segment) + 1):
            result[Cursor(node, offset)] = prefix + node.segment[:offset]
        prefix += node.segment
        for child in (*node.children.values(), *node.silences.values()):
            walk(child, prefix)

    for node in engine.starts.values():
        walk(node, ())
    return result


def state(engine):
    known = paths(engine)

    def place(value):
        if isinstance(value, Cursor):
            return "cursor", known[value]
        if isinstance(value, Window):
            return "window", known[value.child]
        return value

    def partial(value):
        if value is None:
            return None
        return value.steps, value.product, value.count, value.first_position, value.last_time

    return {
        "graph": sorted(map(repr, known.values())),
        "unordered": engine.unordered,
        "snapshot": Counter(
            (place(item.place), partial(item.partial), item.until) for item in engine.snapshot()
        ),
        "misses": Counter(
            (miss.reason, place(miss.place), partial(miss.partial), miss.scenario, miss.need, miss.got)
            for miss in engine.misses
        ),
        "active": engine.active_count,
        "deadline": engine.next_deadline(),
        "position": engine.position,
        "time": engine.time,
        "vocabulary": engine.vocabulary,
    }


def run(engine, operation):
    try:
        return "ok", operation(engine)
    except ValueError as error:
        return "error", str(error)


def compare(seed, scenarios, operations, steps):
    rng = random.Random(seed)
    rules = random_rules(rng, scenarios)
    engine, reference = Engine(rules), ReferenceEngine(rules)
    stream = list(STREAM)
    now, added = 0.0, 0
    for step in range(operations):
        now += rng.choice(steps)
        kind = rng.random()
        if kind < 0.05:
            extra = [f"new{added}"] if rng.random() < 0.5 else []
            scenario = random_scenario(rng, f"n{added}", extra)
            added += 1
            stream.extend(extra)

            def operation(target, scenario=scenario):
                return target.add_scenario(scenario)
        elif kind < 0.2:
            def operation(target, time=now):
                return target.advance(time)
        else:
            event = Event(rng.choice(stream), rng.choice(SCORES), now)

            def operation(target, event=event):
                return target.push(event)
        left, right = run(engine, operation), run(reference, operation)
        assert left == right, (seed, step)
        assert state(engine) == state(reference), (seed, step)


@pytest.mark.parametrize("seed", range(40))
def test_engine_matches_full_traversal_reference(seed):
    compare(seed, 7, 200, [0.0, 0.0, 0.25, 0.5, 1.0, 2.0, 5.0])


@pytest.mark.parametrize("seed", range(8))
def test_dense_stream_matches_full_traversal_reference(seed):
    # много сценариев и мелкий шаг времени: длинные фронты и много окон сразу
    compare(1000 + seed, 12, 400, [0.0, 0.0, 0.05, 0.1, 0.25])
