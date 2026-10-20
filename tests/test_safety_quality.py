"""Качество и безопасность на размеченных запросах."""

from __future__ import annotations

import random

import pytest

from rule_engine.evaluation import (
    build_engine,
    evaluate,
    load_dataset,
    make_typo,
    typo_variants,
)
from rule_engine.policy import Status
from rule_engine.rules import RuleKind


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_dataset_quality_thresholds(engine, dataset):
    metrics = evaluate(engine, dataset).metrics()
    assert metrics["decision_acc"] >= 0.97
    assert metrics["p_at_1"] >= 0.95
    assert metrics["recall_at_3"] >= 0.98
    assert metrics["silent_error"] == 0
    assert metrics["dangerous_silent"] == 0


def test_generated_typos_never_execute_silently(engine, dataset):
    variants = typo_variants(dataset, per_example=3, known_word=engine.lexicon.known_word)
    metrics = evaluate(engine, variants).metrics()
    assert metrics["silent_error"] == 0
    assert metrics["dangerous_silent"] == 0
    assert metrics["recall_at_3"] >= 0.95


@pytest.mark.parametrize(
    "text", ["открой дверь", "сними с охраны", "выключи сигнализацию", "включи духовку", "отопри замок"]
)
def test_dangerous_commands_with_typos_are_never_ok(engine, text):
    rng = random.Random(hash(text) % 1000)
    for _ in range(40):
        words = text.split()
        i = rng.randrange(len(words))
        words[i] = make_typo(words[i], rng)
        noisy = " ".join(words)
        if engine.lexicon.known_word(words[i]) and words[i] != text.split()[i]:
            continue  # опечатка дала другое словарное слово — это другая команда
        result = engine.understand(noisy)
        top = result.decision.top
        if top is not None and top.risk == "high":
            assert result.decision.status is not Status.OK, noisy


def test_backends_make_identical_decisions(dataset):
    decisions = {}
    for backend in ("trie", "ngram", "brute"):
        engine, _ = build_engine(backend=backend)
        decisions[backend] = [engine.understand(ex.query, ex.room).decision.status for ex in dataset]
    assert decisions["trie"] == decisions["ngram"] == decisions["brute"]


def test_safety_ablation_shows_value_of_policy(dataset):
    # без правил безопасности опасные команды выполнялись бы молча
    engine, _ = build_engine(policy_overrides={"confirm_high_risk": False, "fuzzy_max_risk": "high",
                                               "rules_enabled": {"P5-antonym": False}})
    metrics = evaluate(engine, dataset).metrics()
    assert metrics["dangerous_silent"] > 0


def test_confusable_rules_are_generated(ruleset):
    pairs = {(p.alias_a, p.alias_b) for p in ruleset.confusables if p.antonyms}
    assert ("включи", "выключи") in pairs or ("выключи", "включи") in pairs
    assert ruleset.rules_of(RuleKind.CONFUSABLE)


def test_compiler_static_checks(spec):
    from rule_engine.spec import Spec, SpecError

    bad = dict(spec.raw)
    bad["devices"] = [*spec.raw["devices"], {"id": "ghost", "type": "teleporter", "room": "mars"}]
    with pytest.raises(SpecError) as err:
        Spec.from_dict(bad)
    assert any("teleporter" in p for p in err.value.problems)
    assert any("mars" in p for p in err.value.problems)


def test_latency_budget(engine, dataset):
    metrics = evaluate(engine, dataset, repeats=3).metrics()
    assert metrics["p95_ms"] < 25  # с запасом для медленных CI-машин


def test_antonym_stress_no_silent_antonyms_at_default_margin():
    from rule_engine.evaluation import antonym_stress

    rows = {r["antonym_margin"]: r for r in antonym_stress((0.0, 0.7), per_verb=25)}
    assert rows[0.7]["antonym_ok"] == 0
    assert rows[0.7]["correct_ok"] >= 0.9
    # без правила антонимов хотя бы иногда выполнялась бы противоположная команда
    assert rows[0.0]["antonym_ok"] >= rows[0.7]["antonym_ok"]


def test_pymorphy_lemmatizer_gives_same_quality(dataset):
    pytest.importorskip("pymorphy3")
    from rule_engine.engine import Engine

    engine = Engine.from_files(stemmer="pymorphy")
    metrics = evaluate(engine, dataset).metrics()
    assert metrics["silent_error"] == 0 and metrics["decision_acc"] >= 0.97
