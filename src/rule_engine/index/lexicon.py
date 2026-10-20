"""Лексикон: объяснимый индекс синонимов (aliases) команд, устройств и объектов.

Каскад сопоставления слова запроса со словарём (от дешёвого к дорогому):

1. ``EXACT``  — точное совпадение слова (O(1), хэш);
2. ``LAYOUT`` — точное совпадение после исправления раскладки;
3. ``STEM``   — совпадение основы («кухне» ~ «кухня»);
4. ``PREFIX`` — неполное слово («конди» → «кондиционер»);
5. ``FUZZY``  — опечатка, взвешенное расстояние Дамерау–Левенштейна (DP).

Дорогие уровни включаются, только если дешёвые ничего не нашли. Синонимы из
нескольких слов («тёплый пол») собираются из пословных совпадений.

Индекс только дополняется: каждая запись получает номер версии, а откат к
версии ``v`` пересобирает индекс из записей с ``version <= v``. Поэтому
добавление синонима не может испортить старые записи.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from rule_engine.index.fuzzy import FuzzyBackend, make_backend
from rule_engine.text.distance import max_typo_cost
from rule_engine.text.normalize import Token, TokenKind, basic_clean
from rule_engine.text.stem import Stemmer, light_stem


class Kind(StrEnum):
    ACTION = "action"
    DEVICE = "device"
    DEVICE_TYPE = "type"
    ROOM = "room"
    SCENARIO = "scenario"
    OBJECT_TYPE = "object"
    ATTRIBUTE = "attr"


class MatchType(StrEnum):
    EXACT = "exact"
    LAYOUT = "layout"
    STEM = "stem"
    PREFIX = "prefix"
    FUZZY = "fuzzy"


ALL_MATCH_TYPES = frozenset(MatchType)
FUZZY_SLACK = 0.5
STRICT_MATCH_TYPES = frozenset({MatchType.EXACT, MatchType.LAYOUT, MatchType.STEM})
_RANK = {MatchType.EXACT: 0, MatchType.LAYOUT: 1, MatchType.STEM: 2, MatchType.PREFIX: 3, MatchType.FUZZY: 4}


def kind_of(ref: str) -> Kind:
    return Kind(ref.split(":", 1)[0])


@dataclass(frozen=True, slots=True)
class LexEntry:
    id: int
    alias: str
    tokens: tuple[str, ...]
    kind: Kind
    ref: str
    weight: float = 1.0
    source: str = "spec"
    version: int = 0
    hint: str | None = None  # подсказка: тип устройства для глагола «поставь на охрану» и т. п.


@dataclass(frozen=True, slots=True)
class WordMatch:
    word: str  # слово словаря
    match: MatchType
    cost: float  # стоимость правки (0 для точных совпадений)
    quality: float  # 0..1


@dataclass(frozen=True, slots=True)
class SpanMatch:
    start: int
    end: int  # полуинтервал позиций токенов [start, end)
    entry: LexEntry
    words: tuple[WordMatch, ...]

    @property
    def quality(self) -> float:
        return sum(w.quality for w in self.words) / len(self.words)

    @property
    def match(self) -> MatchType:
        """Худший (самый «нечёткий») тип совпадения среди слов синонима."""
        return max((w.match for w in self.words), key=_RANK.__getitem__)

    @property
    def cost(self) -> float:
        return sum(w.cost for w in self.words)

    @property
    def strict(self) -> bool:
        return self.match in STRICT_MATCH_TYPES

    def describe(self, query_tokens: list[Token]) -> str:
        surface = " ".join(t.original for t in query_tokens[self.start : self.end])
        how = self.match.value + (f", d={self.cost:.1f}" if self.cost else "")
        return f"«{surface}» →({how}) «{self.entry.alias}» = {self.entry.ref}"


@dataclass
class AddReport:
    entry: LexEntry | None
    version: int
    duplicate: bool = False
    conflicts: list[LexEntry] = field(default_factory=list)


class Lexicon:
    def __init__(
        self,
        stemmer: Stemmer = light_stem,
        backend: str = "trie",
        match_types: frozenset[MatchType] = ALL_MATCH_TYPES,
        max_alias_len: int = 4,
    ) -> None:
        self.stemmer = stemmer
        self.backend_name = backend
        self.match_types = match_types
        self.max_alias_len = max_alias_len
        self.version = 0
        self.entries: list[LexEntry] = []
        self._next_id = 0
        self.changelog: list[tuple[int, str]] = []
        self._reset_indexes()

    # ------------------------------------------------------------------ build
    def _reset_indexes(self) -> None:
        self._by_alias: dict[str, list[LexEntry]] = defaultdict(list)
        self._by_first: dict[str, list[LexEntry]] = defaultdict(list)
        self._words: set[str] = set()
        self._stem_words: dict[str, set[str]] = defaultdict(set)
        self._fuzzy_keys: dict[str, set[str]] = defaultdict(set)  # слово или основа → слова
        self._fuzzy: FuzzyBackend = make_backend(self.backend_name)
        self._word_cache: dict[tuple[str, bool], list[WordMatch]] = {}

    def _index(self, entry: LexEntry) -> None:
        self._by_alias[entry.alias].append(entry)
        self._by_first[entry.tokens[0]].append(entry)
        for word in entry.tokens:
            if word in self._words:
                continue
            self._words.add(word)
            stem = self.stemmer(word)
            self._stem_words[stem].add(word)
            for key in {word, stem}:
                self._fuzzy_keys[key].add(word)
                self._fuzzy.add(key)
        self._word_cache.clear()

    def normalize_alias(self, alias: str) -> tuple[str, ...]:
        return tuple(basic_clean(alias).split())

    def add(
        self,
        alias: str,
        ref: str,
        *,
        weight: float = 1.0,
        source: str = "spec",
        hint: str | None = None,
        version: int | None = None,
    ) -> AddReport:
        tokens = self.normalize_alias(alias)
        if not tokens:
            raise ValueError(f"empty alias for {ref}")
        if len(tokens) > self.max_alias_len:
            self.max_alias_len = len(tokens)
        text = " ".join(tokens)
        kind = kind_of(ref)
        same = self._by_alias.get(text, [])
        if any(e.ref == ref and e.hint == hint for e in same):
            return AddReport(entry=None, version=self.version, duplicate=True)
        conflicts = [e for e in same if e.kind is kind and e.ref != ref]
        version = self.version if version is None else version
        self.version = max(self.version, version)
        entry = LexEntry(
            id=self._next_id, alias=text, tokens=tokens, kind=kind, ref=ref,
            weight=weight, source=source, version=version, hint=hint,
        )
        self._next_id += 1
        self.entries.append(entry)
        self._index(entry)
        return AddReport(entry=entry, version=self.version, conflicts=conflicts)

    def bump_version(self, note: str) -> int:
        self.version += 1
        self.changelog.append((self.version, note))
        return self.version

    def rollback(self, version: int) -> int:
        """Откат к версии: записи новее ``version`` убираются, индекс пересобирается."""
        if version < 0 or version > self.version:
            raise ValueError(f"no such version: {version}")
        kept = [e for e in self.entries if e.version <= version]
        removed = len(self.entries) - len(kept)
        self.entries = kept
        self.changelog = [c for c in self.changelog if c[0] <= version]
        self.version = version
        self._reset_indexes()
        for entry in self.entries:
            self._index(entry)
        return removed

    # ----------------------------------------------------------------- lookup
    def known_word(self, word: str) -> bool:
        return word in self._words or self.stemmer(word) in self._stem_words

    def refs_for_alias(self, alias: str) -> list[LexEntry]:
        return list(self._by_alias.get(" ".join(self.normalize_alias(alias)), []))

    def match_word(self, token: Token, deep: bool = False) -> list[WordMatch]:
        if token.kind is TokenKind.NUMBER:
            # число может быть частью синонима («привычка 1»), но только точно
            return [WordMatch(token.text, MatchType.EXACT, 0.0, 1.0)] if token.text in self._words else []
        key = (token.text, token.layout_fixed, deep)
        cached = self._word_cache.get(key)
        if cached is not None:
            return cached
        result = self._match_word(token, deep)
        self._word_cache[key] = result
        return result

    def _match_word(self, token: Token, deep: bool = False) -> list[WordMatch]:
        """Каскад для одного слова. ``deep`` — дорогие уровни даже при наличии дешёвых.

        Второй (глубокий) проход нужен, когда дешёвое совпадение не дало ни одного
        синонима целиком: «ноч» совпадает по основе с «ночи» из «спокойной ночи»,
        но пользователь имел в виду «ночник».
        """
        text, enabled = token.text, self.match_types
        strict: list[WordMatch] = []
        if MatchType.EXACT in enabled and text in self._words:
            match = MatchType.LAYOUT if token.layout_fixed else MatchType.EXACT
            strict = [WordMatch(text, match, 0.0, 0.97 if token.layout_fixed else 1.0)]
        elif MatchType.STEM in enabled:
            words = self._stem_words.get(token.stem) or self._stem_words.get(self.stemmer(text))
            if words:
                strict = [WordMatch(w, MatchType.STEM, 0.0, 0.95) for w in sorted(words)]
        if strict and not deep:
            return strict
        # Служебные слова и короткие слова нечётко не ищем: «на» не должно стать «но».
        if token.kind is not TokenKind.WORD or len(text) < 3 or (strict and strict[0].match is not MatchType.STEM):
            return strict
        found: dict[str, WordMatch] = {m.word: m for m in strict}
        if MatchType.FUZZY in enabled:
            # Сначала ищем по слову; по основе — только если слово ничего не дало
            # (короткие основы слишком легко совпадают: «крас» ~ «кран»).
            for query in dict.fromkeys((text, token.stem)):
                budget = max_typo_cost(len(query))
                if budget <= 0 or any(m.match is MatchType.FUZZY for m in found.values()):
                    continue
                for key, cost in self._fuzzy.search(query, budget):
                    for word in self._fuzzy_keys[key]:
                        quality = max(0.3, 0.9 - 0.25 * cost)
                        prev = found.get(word)
                        if prev is None or (prev.match is MatchType.FUZZY and cost < prev.cost):
                            found[word] = WordMatch(word, MatchType.FUZZY, cost, quality)
            # Оставляем только кандидатов, сравнимых с лучшим: при d=1 до «красную»
            # вариант «правую» с d=1.6 — шум, а не альтернатива.
            fuzzy_costs = [m.cost for m in found.values() if m.match is MatchType.FUZZY]
            if fuzzy_costs:
                limit = min(fuzzy_costs) + FUZZY_SLACK
                found = {w: m for w, m in found.items() if m.match is not MatchType.FUZZY or m.cost <= limit}
        if MatchType.PREFIX in enabled and len(text) >= 3:
            for key in self._fuzzy.prefix(text):
                for word in self._fuzzy_keys[key]:
                    if not word.startswith(text):
                        continue
                    quality = 0.6 + 0.3 * len(text) / len(word)
                    prev = found.get(word)
                    if prev is None or (prev.match is not MatchType.STEM and quality > prev.quality):
                        found[word] = WordMatch(word, MatchType.PREFIX, 0.0, quality)
        return sorted(found.values(), key=lambda m: (-m.quality, m.word))

    def match_spans(self, tokens: list[Token]) -> list[SpanMatch]:
        """Все вхождения синонимов в запрос (возможно, пересекающиеся).

        Сначала дешёвый проход; слова, у которых были совпадения, но ни один
        синоним через них не собрался, проходят каскад глубже.
        """
        per_token = [{m.word: m for m in self.match_word(t)} for t in tokens]
        spans = self._spans(tokens, per_token)
        covered = {i for s in spans for i in range(s.start, s.end)}
        retry = [
            i for i, t in enumerate(tokens)
            if i not in covered and t.kind is TokenKind.WORD and len(t.text) >= 3 and per_token[i]
            and all(m.match is MatchType.STEM for m in per_token[i].values())
        ]
        if retry:
            for i in retry:
                per_token[i] = {m.word: m for m in self.match_word(tokens[i], deep=True)}
            spans = self._spans(tokens, per_token)
        return spans

    def _spans(self, tokens: list[Token], per_token: list[dict[str, WordMatch]]) -> list[SpanMatch]:
        spans: list[SpanMatch] = []
        for start, matches in enumerate(per_token):
            for word, first in matches.items():
                for entry in self._by_first.get(word, ()):
                    end = start + len(entry.tokens)
                    if end > len(tokens):
                        continue
                    words = [first]
                    for offset, alias_word in enumerate(entry.tokens[1:], start=1):
                        match = per_token[start + offset].get(alias_word)
                        if match is None:
                            break
                        words.append(match)
                    else:
                        spans.append(SpanMatch(start, end, entry, tuple(words)))
        return spans

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self.entries),
            "aliases": len(self._by_alias),
            "words": len(self._words),
            "fuzzy_keys": len(self._fuzzy),
            "version": self.version,
        }

    @property
    def fuzzy_backend(self) -> FuzzyBackend:
        return self._fuzzy
