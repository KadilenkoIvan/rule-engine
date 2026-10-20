from __future__ import annotations

import pytest

from rule_engine.text.distance import weighted_distance
from rule_engine.text.normalize import Normalizer, TokenKind, basic_clean, fix_layout
from rule_engine.text.stem import light_stem


@pytest.mark.parametrize(
    "forms",
    [
        ("включи", "включить", "включите", "включай"),
        ("выключи", "выключить", "выключите"),
        ("кухня", "кухне", "кухню"),
        ("лампа", "лампу", "лампы"),
        ("красный", "красная", "красную"),
    ],
)
def test_stem_is_consistent_across_forms(forms):
    assert len({light_stem(f) for f in forms}) == 1


def test_stem_never_strips_prefixes():
    # «в-ключи» и «вы-ключи» должны остаться разными основами
    assert light_stem("включи") != light_stem("выключи")
    assert light_stem("открой") != light_stem("закрой")


def test_basic_clean():
    assert basic_clean("  Включи, СВЕТ!!  в Ёлке ") == "включи свет в елке"


def test_layout_fix_only_for_known_words():
    known = {"выключи", "телевизор"}.__contains__
    norm = Normalizer(known_word=known)
    query = norm.normalize("dsrk.xb ntktdbpjh")
    assert [t.text for t in query.tokens] == ["выключи", "телевизор"]
    assert all(t.layout_fixed for t in query.tokens)
    # неизвестное латинское слово не трогаем
    assert Normalizer(known_word=known).normalize("netflix").tokens[0].text == "netflix"
    assert fix_layout("ghbdtn") == "привет"


@pytest.mark.parametrize(
    "text,value",
    [("двадцать два", 22), ("сто двадцать пять", 125), ("пятьдесят", 50), ("22,5", 22.5), ("двпдцать пять", 25)],
)
def test_numerals(text, value):
    tokens = Normalizer().normalize(text).tokens
    numbers = [t for t in tokens if t.kind is TokenKind.NUMBER]
    assert len(numbers) == 1 and numbers[0].value == value


def test_two_separate_numbers():
    tokens = Normalizer().normalize("два три").tokens
    assert [t.value for t in tokens] == [2, 3]


def test_negation_is_kept_not_dropped():
    query = Normalizer().normalize("не включай свет")
    assert query.negated
    assert query.tokens[0].kind is TokenKind.NEGATION


def test_distance_basics():
    assert weighted_distance("включи", "включи") == 0
    assert weighted_distance("включи", "выключи") == 1.0  # одна вставка — антонимы опасно близки
    assert weighted_distance("штроы", "шторы") == pytest.approx(0.7)  # перестановка дешевле двух замен
    assert weighted_distance("торщер", "торшер") == pytest.approx(0.5)  # фонетически близкие ш/щ
    assert weighted_distance("свет", "свер") < 1.0  # т и р — соседние клавиши ЙЦУКЕН
    assert weighted_distance("кондиционер", "свет", max_cost=2) == float("inf")


def test_distance_symmetry_for_unit_costs():
    for a, b in [("кухня", "кухне"), ("пылесос", "пылесрс"), ("дверь", "двкрь")]:
        assert weighted_distance(a, b) == weighted_distance(b, a)
