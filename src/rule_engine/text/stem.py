"""Лёгкий стеммер для русского языка.

Отрезает только окончания (флексии) и возвратные частицы, никогда не трогает
приставки: «в-ключи» и «вы-ключи» остаются разными основами («включ» / «выключ»).
Стеммер применяется одинаково к словарю и к запросу, поэтому важна не
лингвистическая точность, а согласованность ключей.

Если установлен pymorphy3 (extra ``morph``), можно использовать настоящую
лемматизацию через :class:`PymorphyLemmatizer`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

_REFLEXIVE = ("ся", "сь")

# Окончания от длинных к коротким: побеждает самое длинное совпадение.
_ENDINGS = tuple(
    sorted(
        {
            # прилагательные и причастия
            "ого", "его", "ому", "ему", "ыми", "ими", "ая", "яя", "ое", "ее",
            "ые", "ие", "ый", "ий", "ой", "ую", "юю", "ых", "их", "ым", "им",
            # существительные
            "ами", "ями", "ах", "ях", "ов", "ев", "ей", "ам", "ям", "ом", "ем",
            "а", "я", "о", "е", "ы", "и", "у", "ю", "ь",
            # глаголы: инфинитив, повелительное наклонение, личные формы
            "ить", "ать", "ять", "еть", "уть", "ыть", "ти", "ите", "айте", "яйте",
            "ешь", "ет", "ете", "ут", "ют", "ит", "ат", "ят", "ай", "яй", "й",
        },
        key=len,
        reverse=True,
    )
)

MIN_STEM = 3


class Stemmer(Protocol):
    def __call__(self, word: str) -> str: ...


@lru_cache(maxsize=65536)
def light_stem(word: str) -> str:
    """Возвращает основу слова; слова короче 4 символов и числа не меняются."""
    if len(word) < 4 or not word.isalpha():
        return word
    stem = word
    for ending in _REFLEXIVE:
        if stem.endswith(ending) and len(stem) - len(ending) >= MIN_STEM:
            stem = stem[: -len(ending)]
            break
    for ending in _ENDINGS:
        if stem.endswith(ending) and len(stem) - len(ending) >= MIN_STEM:
            return stem[: -len(ending)]
    return stem


class PymorphyLemmatizer:
    """Лемматизация через pymorphy3 (опционально, ``uv sync --extra morph``)."""

    def __init__(self) -> None:
        import pymorphy3

        self._morph = pymorphy3.MorphAnalyzer()
        self._cache: dict[str, str] = {}

    def __call__(self, word: str) -> str:
        cached = self._cache.get(word)
        if cached is None:
            if not word.isalpha():
                cached = word
            else:
                cached = self._morph.parse(word)[0].normal_form.replace("ё", "е")
            self._cache[word] = cached
        return cached


def make_stemmer(kind: str = "auto") -> Stemmer:
    """``light`` — встроенный стеммер, ``pymorphy`` — pymorphy3, ``auto`` — что доступно.

    ``auto`` выбирает встроенный стеммер: он детерминирован и не зависит от версии
    словарей pymorphy; pymorphy нужно запрашивать явно.
    """
    if kind == "pymorphy":
        return PymorphyLemmatizer()
    if kind in ("light", "auto"):
        return light_stem
    raise ValueError(f"unknown stemmer: {kind}")
