"""Бэкенды нечёткого поиска слов.

Все бэкенды возвращают одинаковый результат — слова словаря с взвешенным
расстоянием не больше порога — и отличаются компромиссом скорость/память:

* ``trie``    — префиксное дерево; строки таблицы DP Левенштейна считаются
                один раз для общего префикса и переиспользуются (DP по дереву);
* ``symspell``— индекс удалений: быстрый поиск, но индекс в разы больше;
* ``ngram``   — инвертированный индекс триграмм + проверка кандидатов DP;
* ``brute``   — перебор всего словаря с DP (baseline).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Protocol

from rule_engine.text.distance import initial_row, next_row, weighted_distance


class FuzzyBackend(Protocol):
    name: str

    def add(self, word: str) -> None: ...

    def search(self, word: str, max_cost: float) -> list[tuple[str, float]]: ...

    def prefix(self, word: str, limit: int = 32) -> list[str]: ...

    def __len__(self) -> int: ...


class _TrieNode:
    __slots__ = ("children", "word")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.word: str | None = None


class TrieBackend:
    name = "trie"

    def __init__(self) -> None:
        self.root = _TrieNode()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, _TrieNode())
        if node.word is None:
            node.word = word
            self._size += 1

    def search(self, word: str, max_cost: float) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        first = initial_row(word)
        # Явный стек вместо рекурсии: (узел, символ, строка родителя, строка деда, символ родителя)
        stack = [(child, ch, first, None, None) for ch, child in self.root.children.items()]
        while stack:
            node, ch, prev, prev2, prev_ch = stack.pop()
            row = next_row(word, ch, prev, prev2, prev_ch)
            if node.word is not None and row[-1] <= max_cost:
                results.append((node.word, row[-1]))
            if min(row) <= max_cost:  # иначе ни одно продолжение не уложится в порог
                stack.extend((child, c, row, prev, ch) for c, child in node.children.items())
        return results

    def prefix(self, word: str, limit: int = 32) -> list[str]:
        node = self.root
        for ch in word:
            node = node.children.get(ch)
            if node is None:
                return []
        out: list[str] = []
        stack = [node]
        while stack and len(out) < limit:
            current = stack.pop()
            if current.word is not None and current.word != word:
                out.append(current.word)
            stack.extend(current.children.values())
        return out


class _SortedPrefixMixin:
    """Поиск по префиксу через отсортированный список (для бэкендов без дерева)."""

    _words: set[str]
    _sorted: list[str] | None

    def prefix(self, word: str, limit: int = 32) -> list[str]:
        import bisect

        if self._sorted is None:
            self._sorted = sorted(self._words)
        i = bisect.bisect_left(self._sorted, word)
        out: list[str] = []
        while i < len(self._sorted) and self._sorted[i].startswith(word) and len(out) < limit:
            if self._sorted[i] != word:
                out.append(self._sorted[i])
            i += 1
        return out


class SymSpellBackend(_SortedPrefixMixin):
    name = "symspell"

    def __init__(self, max_deletes: int = 2) -> None:
        self.max_deletes = max_deletes
        self._deletes: dict[str, set[str]] = defaultdict(set)
        self._words: set[str] = set()
        self._sorted = None

    def __len__(self) -> int:
        return len(self._words)

    def _variants(self, word: str, depth: int) -> set[str]:
        out = {word}
        for k in range(1, min(depth, len(word) - 1) + 1):
            for idx in combinations(range(len(word)), k):
                out.add("".join(ch for i, ch in enumerate(word) if i not in idx))
        return out

    def add(self, word: str) -> None:
        if word in self._words:
            return
        self._words.add(word)
        self._sorted = None
        for variant in self._variants(word, self.max_deletes):
            self._deletes[variant].add(word)

    def search(self, word: str, max_cost: float) -> list[tuple[str, float]]:
        candidates: set[str] = set()
        for variant in self._variants(word, self.max_deletes):
            candidates |= self._deletes.get(variant, set())
        out = []
        for cand in candidates:
            d = weighted_distance(word, cand, max_cost)
            if d <= max_cost:
                out.append((cand, d))
        return out


class NgramBackend(_SortedPrefixMixin):
    name = "ngram"

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self._grams: dict[str, set[str]] = defaultdict(set)
        self._words: set[str] = set()
        self._sorted = None

    def __len__(self) -> int:
        return len(self._words)

    def _ngrams(self, word: str) -> set[str]:
        padded = f"{'$' * (self.n - 1)}{word}$"
        return {padded[i : i + self.n] for i in range(len(padded) - self.n + 1)}

    def add(self, word: str) -> None:
        if word in self._words:
            return
        self._words.add(word)
        self._sorted = None
        for gram in self._ngrams(word):
            self._grams[gram].add(word)

    def search(self, word: str, max_cost: float) -> list[tuple[str, float]]:
        grams = self._ngrams(word)
        counts: dict[str, int] = defaultdict(int)
        for gram in grams:
            for cand in self._grams.get(gram, ()):
                counts[cand] += 1
        # Каждая правка портит не больше n n-грамм: оставляем только тех, кто
        # мог уложиться в порог (q-gram lemma), остальных DP не проверяем.
        budget = int(max_cost / 0.5 + 0.999)  # самая дешёвая правка стоит 0.5
        need = max(1, len(grams) - self.n * budget)
        out = []
        for cand, shared in counts.items():
            if shared < need:
                continue
            d = weighted_distance(word, cand, max_cost)
            if d <= max_cost:
                out.append((cand, d))
        return out


class BruteForceBackend(_SortedPrefixMixin):
    name = "brute"

    def __init__(self) -> None:
        self._words: set[str] = set()
        self._sorted = None

    def __len__(self) -> int:
        return len(self._words)

    def add(self, word: str) -> None:
        if word not in self._words:
            self._words.add(word)
            self._sorted = None

    def search(self, word: str, max_cost: float) -> list[tuple[str, float]]:
        out = []
        for cand in self._words:
            d = weighted_distance(word, cand, max_cost)
            if d <= max_cost:
                out.append((cand, d))
        return out


BACKENDS = {
    "trie": TrieBackend,
    "symspell": SymSpellBackend,
    "ngram": NgramBackend,
    "brute": BruteForceBackend,
}


def make_backend(name: str) -> FuzzyBackend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(f"unknown fuzzy backend {name!r}; choose from {sorted(BACKENDS)}") from None
