"""Режим поправок: отложенная выдача, «ой», «не» и маркеры внутри шаблонов (CORRECTIONS.md)."""

import pytest

from rule_engine import Engine, Event, RuleSet

MARKERS = {
    "corrections": [["ой"], ["а", "нет"]],
    "negations": [["не"]],
    "negation_breaks": [["а"], ["но"]],
}


def corrected(make_rules, scenarios, **settings):
    """Правила в режиме поправок: hold 1 с, маркеры из MARKERS; их токены — в словаре."""
    tokens = {token for phrases in MARKERS.values() for phrase in phrases for token in phrase}
    for scenario in scenarios:
        for pattern in scenario["patterns"]:
            tokens.update(step if isinstance(step, str) else step["token"] for step in pattern)
    return make_rules(
        scenarios,
        vocabulary=sorted(tokens),
        **{"self_correction": True, "hold": 1.0, **MARKERS, **settings},
    )


@pytest.mark.parametrize("settings, message", [
    ({"self_correction": 1}, "true или false"),
    ({"hold": -1.0}, "неотрицательным"),
    ({"corrections": [[]]}, "пустая фраза"),
    ({"corrections": [["ой"], ["ой"]]}, "дважды"),
    ({"negations": [["ой"]]}, "и в corrections"),
    ({"negations": [["кофе"]]}, "нет в словаре"),
    ({"negations": ["не"]}, "списком токенов"),
])
def test_invalid_correction_settings_are_rejected(make_rules, settings, message):
    with pytest.raises(ValueError, match=message):
        corrected(make_rules, [{"name": "s", "patterns": [["a"]]}], **settings)


def test_flag_is_required_and_scenario_hold_is_checked(make_rules):
    with pytest.raises(KeyError):
        RuleSet.from_dict({
            "max_gap": 3.0, "min_score": 0.5, "min_confidence": 0.5,
            "vocabulary": ["a"], "scenarios": [{"name": "s", "patterns": [["a"]]}],
        })
    with pytest.raises(ValueError, match="hold"):
        corrected(make_rules, [{"name": "s", "hold": float("inf"), "patterns": [["a"]]}])


def names(matches):
    return [match.scenario for match in matches]


def test_held_match_is_released_after_hold(make_rules):
    engine = Engine(corrected(make_rules, [{"name": "pair", "patterns": [["a", "b"]]}]))
    engine.push(Event("a", 0.9, 0.0))
    assert engine.push(Event("b", 0.9, 0.5)) == []
    assert [held.until for held in engine.held] == [1.5]
    assert engine.next_deadline() == 1.5
    assert engine.active_count == 2  # копия «a» и ждущее срабатывание
    assert engine.advance(1.5) == []  # ровно на сроке ещё ждёт
    [match] = engine.advance(1.6)
    assert (match.scenario, match.released) == ("pair", 1.5)
    assert engine.held == [] and engine.active_count == 1


def test_relative_step_extends_hold_and_foreign_token_does_not(make_rules):
    engine = Engine(corrected(make_rules, [
        {"name": "ab", "patterns": [["a", "b"]]},
        {"name": "ac", "patterns": [["a", "c"]]},
    ]))
    engine.push(Event("a", 0.9, 0.0))
    engine.push(Event("b", 0.9, 0.5))
    engine.push(Event("x", 0.9, 1.0))  # посторонний токен: срок прежний
    assert [held.until for held in engine.held] == [1.5]
    engine.push(Event("c", 0.9, 1.2))  # «a c» начато тем же «a»: шаг родственника
    assert [(held.scenario, held.until) for held in engine.held] == [("ab", 2.2), ("ac", 2.2)]
    assert names(engine.advance(3.0)) == ["ab", "ac"]


def test_hold_zero_and_disabled_mode_emit_at_once(make_rules):
    scenarios = [{"name": "alarm", "hold": 0.0, "patterns": [["a"]]},
                 {"name": "pair", "patterns": [["b", "c"]]}]
    engine = Engine(corrected(make_rules, scenarios))
    assert names(engine.push(Event("a", 0.9, 0.0))) == ["alarm"]
    off = Engine(corrected(make_rules, scenarios, self_correction=False))
    off.push(Event("b", 0.9, 0.0))
    assert names(off.push(Event("c", 0.9, 0.5))) == ["pair"]
    assert off.held == []
