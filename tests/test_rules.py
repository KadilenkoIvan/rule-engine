import pytest

from rule_engine import RuleSet, build_graph


def rules(*scenarios: tuple[str, str]) -> RuleSet:
    return RuleSet.from_dict(
        {
            "vocabulary": ["разогрей", "чайник", "ванну"],
            "scenarios": [{"name": name, "pattern": pattern} for name, pattern in scenarios],
        }
    )


def test_common_start_shares_graph_vertices():
    root = build_graph(rules(("чайник", "разогрей чайник"), ("ванна", "разогрей ванну")))
    assert list(root.children) == ["разогрей"]
    assert set(root.children["разогрей"].children) == {"чайник", "ванну"}


def test_token_outside_vocabulary_in_scenario_raises():
    with pytest.raises(ValueError, match="нет в словаре"):
        rules(("кофе", "свари кофе"))


def test_empty_pattern_raises():
    with pytest.raises(ValueError, match="не содержит токенов"):
        rules(("пусто", "  "))


def test_duplicate_name_raises():
    with pytest.raises(ValueError, match="объявлен дважды"):
        rules(("чайник", "разогрей чайник"), ("чайник", "разогрей ванну"))


def test_same_pattern_in_two_scenarios_raises():
    with pytest.raises(ValueError, match="одну и ту же последовательность"):
        build_graph(rules(("чайник", "разогрей чайник"), ("кипяток", "разогрей чайник")))
