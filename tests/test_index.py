from __future__ import annotations

import random

import pytest

from rule_engine.evaluation import make_typo
from rule_engine.index.fuzzy import BACKENDS
from rule_engine.index.lexicon import Lexicon, MatchType
from rule_engine.text.normalize import Normalizer

WORDS = [
    "включи", "выключи", "телевизор", "телек", "кондиционер", "кондей", "шторы", "штора", "пылесос",
    "чайник", "люстра", "торшер", "ночник", "дверь", "замок", "сигнализация", "кухня", "спальня",
    "гостиная", "прихожая", "манипулятор", "чашка", "кубик", "красный", "синий",
]


@pytest.mark.parametrize("max_cost", [1.0, 1.6])
def test_all_fuzzy_backends_agree_with_brute_force(max_cost):
    rng = random.Random(1)
    backends = {name: cls() for name, cls in BACKENDS.items()}
    for backend in backends.values():
        for word in WORDS:
            backend.add(word)
    queries = [make_typo(rng.choice(WORDS), rng) for _ in range(150)] + WORDS
    for query in queries:
        expected = {w: round(d, 6) for w, d in backends["brute"].search(query, max_cost)}
        for name in ("trie", "ngram"):
            got = {w: round(d, 6) for w, d in backends[name].search(query, max_cost)}
            assert got == expected, (name, query)
        # SymSpell с двумя удалениями гарантирует полноту для ≤2 элементарных правок
        got = {w: round(d, 6) for w, d in backends["symspell"].search(query, max_cost)}
        assert set(got) <= set(expected)
        if max_cost <= 1.0:
            assert got == expected, ("symspell", query)


def test_prefix_search_is_consistent():
    backends = {name: cls() for name, cls in BACKENDS.items()}
    for backend in backends.values():
        for word in WORDS:
            backend.add(word)
    for prefix in ("тел", "конд", "што", "вы"):
        results = {name: sorted(b.prefix(prefix)) for name, b in backends.items()}
        assert len({tuple(r) for r in results.values()}) == 1, results


def _lexicon() -> Lexicon:
    lex = Lexicon()
    lex.add("включи", "action:turn_on")
    lex.add("выключи", "action:turn_off")
    lex.add("кондиционер", "type:ac")
    lex.add("теплый пол", "type:heated_floor")
    lex.add("спокойной ночи", "scenario:good_night")
    lex.add("ночник", "device:bedside_lamp")
    return lex


def _tokens(lex: Lexicon, text: str):
    return Normalizer(lex.stemmer, known_word=lex.known_word).normalize(text).tokens


def test_cascade_levels():
    lex = _lexicon()
    norm = Normalizer(lex.stemmer, known_word=lex.known_word)
    assert lex.match_word(norm.normalize("включи").tokens[0])[0].match is MatchType.EXACT
    assert lex.match_word(norm.normalize("включите").tokens[0])[0].match is MatchType.STEM
    assert lex.match_word(norm.normalize("конди").tokens[0])[0].match in (MatchType.PREFIX, MatchType.STEM)
    fuzzy = lex.match_word(norm.normalize("кондиционр").tokens[0])
    assert fuzzy[0].match is MatchType.FUZZY and fuzzy[0].word == "кондиционер"


def test_multiword_alias_and_deep_second_pass():
    lex = _lexicon()
    spans = lex.match_spans(_tokens(lex, "включи теплый пол"))
    assert any(s.entry.ref == "type:heated_floor" and (s.start, s.end) == (1, 3) for s in spans)
    # «ноч» совпадает по основе с «ночи» (часть сценария), но целиком синоним
    # собрался только через второй, глубокий проход: префикс «ночник».
    spans = lex.match_spans(_tokens(lex, "включи ноч"))
    assert any(s.entry.ref == "device:bedside_lamp" for s in spans)


def test_add_alias_is_versioned_and_rollback_keeps_old_entries():
    lex = _lexicon()
    before = [(e.alias, e.ref) for e in lex.entries]
    v1 = lex.version + 1
    report = lex.add("кондер", "type:ac", version=v1)
    assert report.entry is not None and report.entry.version == v1
    assert lex.add("кондер", "type:ac", version=v1 + 1).duplicate
    conflict = lex.add("включи", "action:turn_off", version=v1 + 1)
    assert [c.ref for c in conflict.conflicts] == ["action:turn_on"]
    assert lex.version == v1 + 1
    removed = lex.rollback(0)
    assert removed == 2
    assert [(e.alias, e.ref) for e in lex.entries] == before
    assert not lex.refs_for_alias("кондер")
