import pytest

from rule_engine import RuleSet


@pytest.fixture
def make_rules():
    """Правила с явными порогами; словарь по умолчанию собран из шаблонов."""
    def make(scenarios, *, vocabulary=None, **settings):
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
            "self_correction": False,
            "max_gap": 3.0,
            "min_score": 0.5,
            "min_confidence": 0.5,
            **settings,
            "vocabulary": vocabulary,
            "scenarios": scenarios,
        })

    return make
