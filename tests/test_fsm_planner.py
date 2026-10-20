from __future__ import annotations

import itertools
import math
import random

import pytest

from rule_engine.engine import REQUEST_FSM
from rule_engine.fsm import Edge, FSMDefinition, FSMInstance, TransitionError, plan_path
from rule_engine.planner import Step, allocate_attempts
from rule_engine.policy import Status


def _graph() -> FSMDefinition:
    return FSMDefinition(
        "g",
        ["a", "b", "c", "d"],
        "a",
        [
            Edge("x", "a", "b", cost=1),
            Edge("y", "b", "c", cost=1),
            Edge("z", "a", "c", cost=5),
            Edge("w", "c", "d", cost=1, guard=lambda ctx: ctx.get("ok", False), guard_name="ok"),
        ],
    )


def test_plan_path_is_cheapest():
    path = plan_path(_graph(), "a", lambda s: s == "c")
    assert [e.event for e in path] == ["x", "y"]


def test_plan_path_respects_step_limit_and_guards():
    fsm = _graph()
    assert [e.event for e in plan_path(fsm, "a", lambda s: s == "c", max_steps=1)] == ["z"]
    assert plan_path(fsm, "a", lambda s: s == "d") is None
    assert [e.event for e in plan_path(fsm, "a", lambda s: s == "d", {"ok": True})] == ["x", "y", "w"]
    assert plan_path(fsm, "a", lambda s: s == "a") == []


def test_fsm_instance_rejects_invalid_edges():
    machine = FSMInstance(_graph())
    with pytest.raises(TransitionError):
        machine.fire("y")
    machine.fire("x")
    machine.fire("y")
    with pytest.raises(TransitionError, match="ok"):
        machine.fire("w", {"ok": False})
    assert [h.event for h in machine.history] == ["x", "y"]


def test_request_pipeline_cannot_skip_stages():
    request = FSMInstance(REQUEST_FSM)
    with pytest.raises(TransitionError):
        request.fire("execute")
    for event in ("normalize", "parse", "resolve", "plan", "execute", "succeed"):
        request.fire(event)
    assert request.finished


def test_planner_idempotent_noop(engine):
    engine.handle("включи свет на кухне")
    plan = engine.planner.plan(engine.understand("включи свет на кухне").decision.chosen, engine.world)
    assert not plan.steps and plan.noops


def test_planner_inserts_auto_steps(engine):
    decision = engine.understand("возьми красную чашку").decision
    plan = engine.planner.plan(decision.chosen, engine.world)
    assert [(s.action, s.auto) for s in plan.steps] == [("turn_on", True), ("grasp", False)]


def test_planner_precondition_failure(engine):
    # шарик виден с уверенностью 0.42 < 0.5 — guard object_visible не пускает
    decision = engine.understand("возьми мяч").decision
    plan = engine.planner.plan(decision.chosen, engine.world)
    assert plan.errors and plan.errors[0].code.value == "PRECONDITION_FAILED"


def test_set_value_on_off_device_turns_it_on_first(engine):
    decision = engine.understand("громкость телевизора 30").decision
    assert decision.status is Status.OK
    plan = engine.planner.plan(decision.chosen, engine.world)
    assert [s.action for s in plan.steps] == ["turn_on", "set_volume"]


def _brute_force_attempts(steps, probs, deadline, max_attempts, backoff, factor, quantum=50.0):
    best, best_choice = -math.inf, None
    for choice in itertools.product(range(1, max_attempts + 1), repeat=len(steps)):
        total = 0
        logp = 0.0
        for step, p, k in zip(steps, probs, choice):
            cost = step.latency_ms + sum(step.timeout_ms + backoff * factor ** (j - 1) for j in range(1, k))
            total += math.ceil(cost / quantum)
            logp += math.log(1 - (1 - p) ** k)
        if total <= deadline // quantum and logp > best + 1e-12:
            best, best_choice = logp, choice
    return best_choice, best


@pytest.mark.parametrize("seed", range(12))
def test_attempt_allocation_dp_is_optimal(seed, ruleset):
    rng = random.Random(seed)
    edge = Edge("turn_on", "off", "on")
    steps = [
        Step("d", "turn_on", "off", "on", edge, timeout_ms=rng.choice([300, 800, 1500]),
             latency_ms=rng.choice([50, 150, 400]))
        for _ in range(rng.randint(1, 4))
    ]
    probs = [rng.uniform(0.5, 0.99) for _ in steps]
    deadline = rng.choice([1000, 2500, 5000])
    attempts, p_all = allocate_attempts(steps, probs, deadline, 4, 100, 2.0)
    expected, best = _brute_force_attempts(steps, probs, deadline, 4, 100, 2.0)
    if expected is None:
        assert attempts == [1] * len(steps)
    else:
        assert math.isclose(math.log(p_all), best, rel_tol=1e-9)
