"""Engine benchmarks: event time, build, snapshots, add_scenario and memory.

uv run python benchmarks/engine.py [workload ...] [--memory]

Time is the best of REPEATS runs by total event time; median and percentiles are of that run.
Snapshots are timed apart from events. `--memory` is a separate run under tracemalloc.
"""

import gc
import json
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

from rule_engine import Engine, Event, RuleSet
from rule_engine.application.stream import parse_stream

REPEATS = 3
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def rules(scenarios, vocabulary=None, **settings):
    """Rules of a workload; the vocabulary defaults to the tokens of the patterns."""
    if vocabulary is None:
        tokens = set()
        for scenario in scenarios:
            for pattern in scenario["patterns"]:
                for step in pattern:
                    if isinstance(step, str):
                        tokens.add(step)
                    elif "absent" in step:
                        tokens.update(step["absent"])
                    else:
                        tokens.add(step["token"])
        vocabulary = sorted(tokens - {"any", "any_vac"})
    return RuleSet.from_dict({
        "self_correction": False, "max_gap": 3.0, "min_score": 0.3, "min_confidence": 0.3,
        **settings,
        "vocabulary": vocabulary, "scenarios": scenarios,
    })


# Нагрузка: (правила, операции). Операция — Event, float (advance) или ("add", dict).

def chains(busy):
    """1. Long compressed chains: 40 scenarios of 40 steps without common prefixes."""
    rng = random.Random(1)
    scenarios = [{"name": f"c{i}", "patterns": [[f"c{i}_{k}" for k in range(40)]],
                  "max_gap": 1e9} for i in range(40)]
    ops, now = [], 0.0
    for n in range(5000):
        now += 0.01
        if busy:
            # много занятых позиций: каждая цепочка продвигается, старты повторяются
            i = rng.randrange(40)
            k = rng.randrange(40) if rng.random() < 0.8 else 0
            ops.append(Event(f"c{i}_{k}", 0.9, now))
        else:
            # мало занятых: в основном посторонние токены
            token = f"c{rng.randrange(40)}_0" if rng.random() < 0.01 else "noise"
            ops.append(Event(token, 0.9, now))
    return rules(scenarios), ops


def wide():
    """2. Wide graph: 800 scenarios, many waiting matches, fitting events are rare."""
    rng = random.Random(2)
    vocab = [f"t{i}" for i in range(200)]
    scenarios = [{"name": f"w{i}", "patterns": [[rng.choice(vocab) for _ in range(rng.randint(3, 6))]],
                  "max_gap": 60.0} for i in range(800)]
    ops, now = [], 0.0
    for n in range(4000):
        now += 0.05
        token = rng.choice(vocab) if n < 1000 or rng.random() < 0.02 else f"noise{rng.randrange(50)}"
        ops.append(Event(token, rng.choice([0.6, 0.8, 0.95]), now))
    return rules(scenarios, vocabulary=vocab + [f"noise{i}" for i in range(50)]), ops


def incomparable():
    """3. Many incomparable candidates in one place: start scores decrease."""
    scenarios = [{"name": "s", "patterns": [["a", "b", "c"]], "max_gap": 1e9}]
    ops, now = [], 0.0
    for n in range(400):
        now += 0.01
        ops.append(Event("a", 0.99 - n * 0.002, now))
    rng = random.Random(3)
    for n in range(2000):
        now += 0.01
        ops.append(Event("b" if rng.random() < 0.05 else "z", 0.9, now))
    return rules(scenarios, min_confidence=0.0, min_score=0.0), ops


def unordered():
    """4. `any` patterns with repeats and wildcard steps, various thresholds."""
    rng = random.Random(4)
    vocab = [f"u{i}" for i in range(30)]
    scenarios = []
    for i in range(100):
        pattern = []
        for _ in range(rng.randint(3, 5)):
            r = rng.random()
            token = "any" if r < 0.1 else "any_vac" if r < 0.2 else rng.choice(vocab[:10])
            pattern.append({"token": token, "min_score": rng.choice([0.3, 0.6])})
        scenarios.append({"name": f"u{i}", "order": "any", "patterns": [pattern],
                          "max_gap": rng.choice([1.0, 3.0])})
    ops, now = [], 0.0
    for n in range(5000):
        now += 0.2
        token = rng.choice(vocab) if rng.random() < 0.9 else "unknown"
        ops.append(Event(token, rng.choice([0.4, 0.7, 0.9]), now))
    return rules(scenarios, vocabulary=vocab), ops


def silences():
    """5. Many silence windows: `before`, single and wildcard forbidden tokens."""
    rng = random.Random(5)
    vocab = [f"s{i}" for i in range(40)]
    scenarios = []
    for i in range(200):
        r = rng.random()
        absent = ["any"] if r < 0.15 else ["any_vac"] if r < 0.3 else [rng.choice(vocab)]
        silence = {"absent": absent, "for": rng.choice([0.5, 2.0, 10.0]),
                   "before": rng.choice([0.0, 0.0, 1.0, 5.0])}
        scenarios.append({"name": f"h{i}", "patterns": [[rng.choice(vocab), silence, rng.choice(vocab)]],
                          "max_gap": 5.0})
    ops, now = [], 0.0
    for n in range(5000):
        now += rng.choice([0.05, 0.3, 1.0])
        if n % 10 == 0:
            ops.append(now)
            continue
        ops.append(Event(rng.choice(vocab), rng.choice([0.5, 0.9]), now))
    return rules(scenarios, vocabulary=vocab), ops


def scenario_key(item):
    """Key of the scenario's pattern: scenarios with equal keys conflict."""
    pattern = item["patterns"][0]
    if item.get("order") == "any":
        return "any", json.dumps(sorted(pattern))
    return "sequence", json.dumps(pattern)


def mixed():
    """6. Mixed load and dynamic inserts with splits."""
    rng = random.Random(6)
    vocab = [f"m{i}" for i in range(60)]
    ops, now, seen, count = [], 0.0, set(), 0
    initial = []
    while len(initial) < 200:
        item = _mixed_scenario(rng, vocab, count)
        count += 1
        key = scenario_key(item)
        if key not in seen:
            seen.add(key)
            initial.append(item)
    for n in range(5000):
        now += rng.choice([0.1, 0.5])
        if n % 50 == 49:
            item = _mixed_scenario(rng, vocab, count)
            count += 1
            key = scenario_key(item)
            if key not in seen:
                seen.add(key)
                ops.append(("add", item))
            continue
        ops.append(Event(rng.choice(vocab[:20]), rng.choice([0.6, 0.9]), now))
    return rules(initial, vocabulary=vocab), ops


def _mixed_scenario(rng, vocab, i):
    """Scenario number `i` of the mixed workload: every 5th is `any`, every 7th has a silence."""
    if i % 5 == 0:
        return {"name": f"x{i}", "order": "any",
                "patterns": [[rng.choice(vocab[:15]) for _ in range(3)]]}
    pattern = [rng.choice(vocab[:8]) for _ in range(rng.randint(2, 8))]
    if i % 7 == 0:
        pattern.insert(rng.randrange(1, len(pattern)), {"absent": [rng.choice(vocab)], "for": 1.0})
    return {"name": f"x{i}", "patterns": [pattern], "max_gap": 4.0}


def small():
    """7. Examples: smart_home.json and all examples/*.txt streams in a row, shifted in time."""
    data = RuleSet.from_dict(json.loads((EXAMPLES / "smart_home.json").read_text()))
    ops, offset = [], 0.0
    for _ in range(20):
        last = 0.0
        for path in sorted(EXAMPLES.glob("*.txt")):
            for item in parse_stream(path.read_text().splitlines()):
                if isinstance(item, Event):
                    if item.token not in data.vocabulary:
                        continue
                    ops.append(Event(item.token, item.score, item.time + offset))
                    last = max(last, item.time)
                else:
                    ops.append(item + offset)
                    last = max(last, item)
            offset += last + 100.0
    return data, ops


WORKLOADS = {
    "1a-chains-sparse": lambda: chains(False),
    "1b-chains-busy": lambda: chains(True),
    "2-wide": wide,
    "3-incomparable": incomparable,
    "4-unordered": unordered,
    "5-silences": silences,
    "6-mixed-dynamic": mixed,
    "7-examples": small,
}


def run(engine, ops, timing):
    """Runs the operations; with `timing`, also times a snapshot every 500 operations.

    Returns:
        Times of events and advances, times of add_scenario, total snapshot time; all in ns.
    """
    push_ns, add_ns = [], []
    snapshot_ns = 0
    clock = time.perf_counter_ns
    for index, op in enumerate(ops):
        if isinstance(op, Event):
            start = clock()
            engine.push(op)
            push_ns.append(clock() - start)
        elif isinstance(op, tuple):
            start = clock()
            engine.add_scenario(op[1])
            add_ns.append(clock() - start)
        else:
            start = clock()
            engine.advance(op)
            push_ns.append(clock() - start)
        if timing and index % 500 == 0:
            start = clock()
            engine.snapshot()
            snapshot_ns += clock() - start
    return push_ns, add_ns, snapshot_ns


def quantile(values, q):
    """Quantile `q` of the values."""
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    """Runs the workloads named in the arguments, all by default."""
    names = [name for name in sys.argv[1:] if not name.startswith("--")] or list(WORKLOADS)
    memory = "--memory" in sys.argv
    for name in names:
        rule_set, ops = WORKLOADS[name]()
        if memory:
            measure_memory(name, rule_set, ops)
        else:
            measure_time(name, rule_set, ops)


def measure_memory(name, rule_set, ops):
    """Prints memory after the build, the peak under load and at the end."""
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    engine = Engine(rule_set)
    built = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.reset_peak()
    run(engine, ops, False)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{name:18} память: после построения {built / 1024:8.0f} КиБ, "
          f"пик {(peak - base) / 1024:8.0f} КиБ, в конце {(current - base) / 1024:8.0f} КиБ, "
          f"активных {engine.active_count}", flush=True)


def measure_time(name, rule_set, ops):
    """Prints event time quantiles, build and snapshot time of the best of REPEATS runs."""
    best = None
    for _ in range(REPEATS):
        gc.collect()
        start = time.perf_counter_ns()
        engine = Engine(rule_set)
        build = time.perf_counter_ns() - start
        push_ns, add_ns, snapshot_ns = run(engine, ops, True)
        total = sum(push_ns)
        if best is None or total < best[0]:
            best = (total, build, push_ns, add_ns, snapshot_ns, engine.active_count)
    total, build, push_ns, add_ns, snapshot_ns, active = best
    line = (f"{name:18} событие мкс: медиана {statistics.median(push_ns) / 1e3:8.1f} "
            f"p95 {quantile(push_ns, 0.95) / 1e3:8.1f} p99 {quantile(push_ns, 0.99) / 1e3:8.1f} "
            f"среднее {total / len(push_ns) / 1e3:8.1f} | построение {build / 1e6:6.1f} мс "
            f"| снимки {snapshot_ns / 1e6:7.1f} мс | активных в конце {active}")
    if add_ns:
        line += f" | add_scenario мкс: медиана {statistics.median(add_ns) / 1e3:.0f} макс {max(add_ns) / 1e3:.0f}"
    print(line, flush=True)


if __name__ == "__main__":
    main()
