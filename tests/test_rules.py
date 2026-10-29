import pytest

from rule_engine import RuleSet, Start, build_graph


def rules(*scenarios: tuple[str, str]) -> RuleSet:
    return RuleSet.from_dict(
        {
            "self_correction": False,
            "max_gap": 3.0,
            "min_score": 0.5,
            "min_confidence": 0.6,
            "vocabulary": ["разогрей", "чайник", "ванну"],
            "scenarios": [{"name": name, "patterns": [pattern.split()]} for name, pattern in scenarios],
        }
    )


def test_common_start_shares_graph_vertices():
    root = build_graph(rules(("чайник", "разогрей чайник"), ("ванна", "разогрей ванну")))
    assert list(root) == [Start("разогрей", 0.5)]
    [node] = root.values()
    assert [step.token for step in node.segment] == ["разогрей"]
    assert {edge.token for edge in node.children} == {"чайник", "ванну"}


def test_token_outside_vocabulary_in_scenario_raises():
    with pytest.raises(ValueError, match="нет в словаре"):
        rules(("кофе", "свари кофе"))


def test_empty_pattern_raises():
    with pytest.raises(ValueError, match="пустой шаблон"):
        rules(("пусто", "  "))


def test_duplicate_name_raises():
    with pytest.raises(ValueError, match="объявлен дважды"):
        rules(("чайник", "разогрей чайник"), ("чайник", "разогрей ванну"))


def test_same_pattern_in_two_scenarios_raises():
    with pytest.raises(ValueError, match="одну и ту же последовательность"):
        build_graph(rules(("чайник", "разогрей чайник"), ("кипяток", "разогрей чайник")))


@pytest.mark.parametrize("wildcard", ["any", "any_vac"])
def test_reserved_tokens_cannot_be_in_vocabulary(make_rules, wildcard):
    with pytest.raises(ValueError, match="зарезервированы"):
        make_rules([{"name": "s", "patterns": [[wildcard]]}], vocabulary=[wildcard])


@pytest.mark.parametrize("pattern, order, message", [
    ([{"absent": ["x"], "for": 1.0}, "a"], "sequence", "первым"),
    (["a", {"absent": ["x"], "for": 1.0}, {"absent": ["x"], "for": 1.0}], "sequence", "сразу после"),
    (["a", {"absent": ["x"], "for": 1.0}], "any", "только при order"),
    (["a", {"absent": [], "for": 1.0}], "sequence", "пустой список"),
    (["a", {"absent": ["x", "x"], "for": 1.0}], "sequence", "повторяются"),
    (["a", {"absent": ["x"], "for": 0.0}], "sequence", "положительным"),
    (["a", {"absent": ["x"], "for": 1.0, "before": -1.0}], "sequence", "неотрицательным"),
    (["a", {"absent": ["x"], "for": 1.0, "before": float("inf")}], "sequence", "конечным"),
    (["a", {"token": "b", "max_gap": 1.0}], "any", "не задаётся"),
    ([{"token": "a", "max_gap": 1.0}], "sequence", "первого шага"),
])
def test_invalid_steps_are_rejected(make_rules, pattern, order, message):
    with pytest.raises(ValueError, match=message):
        make_rules([{"name": "s", "order": order, "patterns": [pattern]}])


@pytest.mark.parametrize("settings", [
    {"unexpected": True}, {"max_gap": 0.0}, {"min_score": -0.1}, {"min_confidence": 1.1},
])
def test_invalid_root_settings_are_rejected(make_rules, settings):
    with pytest.raises(ValueError):
        make_rules([{"name": "s", "patterns": [["a"]]}], **settings)
