"""Взвешенное расстояние Дамерау–Левенштейна (вариант OSA) — динамическое программирование.

D[i][j] — минимальная стоимость превращения a[:i] в b[:j]:

    D[i][j] = min(D[i-1][j]   + del,
                  D[i][j-1]   + ins,
                  D[i-1][j-1] + sub(a[i-1], b[j-1]),
                  D[i-2][j-2] + transpose        # если a[i-2:i] == b[j-1:j-3:-1])

Стоимость замены зависит от того, насколько замена «вероятна» как опечатка:
соседние клавиши ЙЦУКЕН и фонетически близкие буквы стоят дешевле.
Расчёт ведётся по строкам с ранним выходом, когда минимум строки превысил
порог (дальше расстояние может только расти).
"""

from __future__ import annotations

from functools import lru_cache

INSERT = 1.0
DELETE = 1.0
SUBSTITUTE = 1.0
TRANSPOSE = 0.7
ADJACENT_KEY = 0.6
PHONETIC = 0.5

_ROWS_RU = ("йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю")
_ROWS_EN = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_PHONETIC_PAIRS = (
    "ао", "еи", "еэ", "иы", "шщ", "ьъ", "зс", "дт", "бп", "гк", "вф", "жш", "йи", "ея", "юу",
)


def _adjacency(rows: tuple[str, ...]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            neighbours = [(r, c - 1), (r, c + 1), (r - 1, c), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c)]
            for rr, cc in neighbours:
                if 0 <= rr < len(rows) and 0 <= cc < len(rows[rr]):
                    pairs.add(frozenset((ch, rows[rr][cc])))
    return pairs


_ADJACENT = _adjacency(_ROWS_RU) | _adjacency(_ROWS_EN)
_PHONETIC = {frozenset(p) for p in _PHONETIC_PAIRS}


@lru_cache(maxsize=4096)
def substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    pair = frozenset((a, b))
    if pair in _PHONETIC:
        return PHONETIC
    if pair in _ADJACENT:
        return ADJACENT_KEY
    return SUBSTITUTE


def next_row(
    word: str,
    ch: str,
    prev: list[float],
    prev2: list[float] | None,
    prev_ch: str | None,
) -> list[float]:
    """Одна строка таблицы DP: добавили к префиксу словаря символ ``ch``.

    ``prev`` — строка для префикса без ``ch``, ``prev2`` — ещё на символ короче
    (нужна для транспозиции). Используется и в обычном расчёте, и при обходе
    префиксного дерева, где строки переиспользуются для общих префиксов.
    """
    row = [prev[0] + INSERT]
    for j in range(1, len(word) + 1):
        wc = word[j - 1]
        cost = min(
            prev[j] + INSERT,
            row[j - 1] + DELETE,
            prev[j - 1] + substitution_cost(wc, ch),
        )
        if (
            prev2 is not None
            and prev_ch is not None
            and j > 1
            and wc == prev_ch
            and word[j - 2] == ch
            and wc != ch
        ):
            cost = min(cost, prev2[j - 2] + TRANSPOSE)
        row.append(cost)
    return row


def initial_row(word: str) -> list[float]:
    return [j * DELETE for j in range(len(word) + 1)]


def weighted_distance(a: str, b: str, max_cost: float = float("inf")) -> float:
    """Расстояние от запроса ``a`` до словарного слова ``b``.

    Возвращает ``inf``, если расстояние гарантированно больше ``max_cost``.
    """
    if a == b:
        return 0.0
    if abs(len(a) - len(b)) * min(INSERT, DELETE) > max_cost:
        return float("inf")
    prev2: list[float] | None = None
    prev = initial_row(a)
    prev_ch: str | None = None
    for ch in b:
        row = next_row(a, ch, prev, prev2, prev_ch)
        if min(row) > max_cost:
            return float("inf")
        prev2, prev, prev_ch = prev, row, ch
    result = prev[-1]
    return result if result <= max_cost else float("inf")


def max_typo_cost(length: int) -> float:
    """Сколько ошибок допускаем в слове данной длины."""
    if length <= 3:
        return 0.0
    if length <= 5:
        return 1.0
    if length <= 8:
        return 1.6
    return 2.0
