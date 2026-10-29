"""Step tokens for interfaces: labels of `any` and `any_vac` and the fit rule.

The rule is the domain's: the web finds where a match ended by it and keeps no copy.
"""

from rule_engine.domain.rules import ANY, ANY_VOCABULARY, fits

LABELS = {ANY: "любой токен", ANY_VOCABULARY: "любой токен словаря"}

__all__ = ["fits", "token_label"]


def token_label(token: str) -> str:
    """Label of a step token: `any` and `any_vac` in words, others as they are."""
    return LABELS.get(token, token)
