import random

import pytest

from rule_engine import Cursor, Engine, Event, Start
from rule_engine.domain.engine import Window
from rule_engine.domain.graph import Node, Silence, attach, branches, edge_of


def expanded_graph(rules):
    """Эталон без radix-вставки: отдельная вершина на каждый шаг шаблона."""
    starts = {}
    for scenario in rules.scenarios:
        if scenario.order != "sequence":
            continue
        for first, *rest in scenario.patterns:
            edges = [Start(first.token, rules.threshold(scenario, first))]
            edges.extend(edge_of(rules, scenario, step) for step in rest)
            parent = None
            nodes = []
            for edge in edges:
                # Вид ребра определяет, в каком словаре родителя лежит потомок.
                children = starts if parent is None else branches(parent, edge)
                node = children.get(edge)
                if node is None:
                    node = Node((edge,))
                    if parent is None:
                        starts[edge] = node
                    else:
                        attach(parent, edge, node)
                nodes.append(node)
                parent = node
            node.scenario = scenario.name
            target = (sum(not isinstance(edge, Silence) for edge in edges),
                      rules.confidence_threshold(scenario))
            for node in nodes:
                node.through.add(target)
            for node in nodes[:-1]:
                node.ahead.add(target)
    return starts


def engine_state(engine):
    """Сравнение логики по путям доменного графа, без интерфейсной проекции."""
    paths = {}

    def walk(node, prefix):
        for offset in range(1, len(node.segment) + 1):
            paths[Cursor(node, offset)] = prefix + node.segment[:offset]
        prefix += node.segment
        for child in (*node.children.values(), *node.silences.values()):
            walk(child, prefix)

    for node in engine.starts.values():
        walk(node, ())

    def place(value):
        if isinstance(value, Cursor):
            return ("cursor", paths[value])
        if isinstance(value, Window):
            return ("window", paths[value.child])
        return value

    def partial(value):
        if value is None:
            return None
        return (value.steps, value.product, value.count, value.first_position, value.last_time)

    return (
        [(place(w.place), partial(w.partial), w.until) for w in engine.snapshot()],
        [(m.reason, place(m.place), partial(m.partial), m.scenario, m.need, m.got)
         for m in engine.misses],
    )


@pytest.mark.parametrize("seed", range(12))
def test_compressed_and_expanded_graphs_have_identical_traces(make_rules, seed):
    rng = random.Random(seed)
    scenarios = []
    for i in range(8):
        order = "any" if i % 3 == 0 else "sequence"
        pattern = [rng.choice(["a", "b", "any", "any_vac"])]
        for _ in range(3):
            if order == "sequence" and rng.random() < 0.4:
                pattern.append({"absent": [rng.choice(["x", "any", "any_vac"])],
                                "for": rng.choice([0.5, 1.0, 3.0]),
                                "before": rng.choice([0.0, 1.0, 4.0])})
            pattern.append(rng.choice(["a", "b", "c", "any", "any_vac"]))
        pattern.append(f"end{i}")
        scenarios.append({"name": str(i), "order": order, "patterns": [pattern],
                          "max_gap": rng.choice([1.0, 3.0, 8.0]),
                          "min_confidence": rng.choice([0.3, 0.6, 0.8]),
                          "cooldown": rng.choice([0.0, 4.0])})
    rules = make_rules(scenarios)
    compressed, expanded = Engine(rules), Engine(rules)
    expanded._starts = expanded_graph(rules)
    now = 0.0
    for operation in range(120):
        now += rng.choice([0.0, 0.5, 1.0, 3.0, 8.0])
        if operation % 5 == 0:
            left, right = compressed.advance(now), expanded.advance(now)
        else:
            event = Event(rng.choice(["a", "b", "c", "x", "unknown"] +
                                     [f"end{i}" for i in range(8)]),
                          rng.choice([0.0, 0.4, 0.5, 0.8, 1.0]), now)
            left, right = compressed.push(event), expanded.push(event)
        assert left == right, (seed, operation)
        assert engine_state(compressed) == engine_state(expanded), (seed, operation)
        assert compressed.next_deadline() == expanded.next_deadline()
        assert compressed.active_count == expanded.active_count
        assert compressed.position == expanded.position
