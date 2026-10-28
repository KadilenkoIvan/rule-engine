"""Токены шагов для интерфейсов: подписи `any` и `any_vac` и правило «токен подходит к шагу».

Правило одно, из домена: веб ищет по нему место срабатывания и не держит своей копии.
"""

from rule_engine.domain.rules import ANY, ANY_VOCABULARY, fits

LABELS = {ANY: "любой токен", ANY_VOCABULARY: "любой токен словаря"}

__all__ = ["fits", "token_label"]


def token_label(token: str) -> str:
    """Подпись токена шага: `any` и `any_vac` словами, остальные как есть."""
    return LABELS.get(token, token)
