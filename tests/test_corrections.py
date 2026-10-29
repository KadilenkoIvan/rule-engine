"""Режим поправок: отложенная выдача, «ой», «не» и маркеры внутри шаблонов (CORRECTIONS.md)."""

import pytest

from rule_engine import RuleSet

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
