"""Нормализация запроса.

Разрешённые преобразования (не меняют смысл команды):
  * Unicode NFKC, нижний регистр, ``ё → е``;
  * удаление пунктуации и схлопывание пробелов;
  * исправление раскладки (``dsrk.xb → выключи``), только если результат есть в словаре;
  * слова-числительные → числа («двадцать два» → 22);
  * основа слова (стемминг) хранится *рядом* с исходным словом, а не вместо него.

Запрещённые преобразования (меняют смысл, поэтому не выполняются):
  * удаление отрицаний («не», «нельзя») — они размечаются отдельно;
  * удаление чисел и единиц измерения;
  * удаление приставок («в-ключи» / «вы-ключи»);
  * агрессивное слияние разных глаголов в одну основу.

Каждое применённое преобразование записывается в ``trace`` для объяснения.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from rule_engine.text.distance import weighted_distance
from rule_engine.text.stem import Stemmer, light_stem


class TokenKind(StrEnum):
    WORD = "word"
    NUMBER = "number"
    NEGATION = "negation"
    CONJUNCTION = "conjunction"
    QUANTIFIER = "quantifier"
    FILLER = "filler"


@dataclass(frozen=True, slots=True)
class Token:
    text: str  # нормализованная форма
    stem: str
    kind: TokenKind
    original: str  # как было во входной строке
    value: float | None = None  # для чисел
    layout_fixed: bool = False


@dataclass
class NormalizedQuery:
    original: str
    tokens: list[Token]
    trace: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def content_tokens(self) -> list[Token]:
        return [t for t in self.tokens if t.kind in (TokenKind.WORD, TokenKind.NUMBER)]

    @property
    def negated(self) -> bool:
        return any(t.kind is TokenKind.NEGATION for t in self.tokens)


NEGATIONS = frozenset({"не", "нельзя", "никогда"})
CONJUNCTIONS = frozenset({"и", "а", "потом", "затем", "также", "еще", "плюс"})
QUANTIFIERS = frozenset({"все", "весь", "всю", "всех", "везде", "всем", "целиком"})
FILLERS = frozenset(
    {
        "пожалуйста", "плиз", "please", "в", "во", "на", "у", "с", "со", "к", "ко",
        "по", "из", "от", "до", "для", "мне", "нам", "мой", "моя", "мое", "мою",
        "давай", "можешь", "можно", "ты", "вы", "ну", "это", "эту", "этот", "тот",
        "ту", "там", "тут", "здесь", "сейчас", "срочно", "быстро", "алиса", "робот",
        "эй", "слушай", "окей", "ок", "сделай", "поставь", "установи", "выставь",
        "градус", "градуса", "градусов", "процент", "процента", "процентов", "уровень",
        "свой", "свою", "же", "бы", "чтобы", "надо", "нужно", "хочу", "пусть", "мы",
        "который", "которая", "которую", "то",
    }
)

_EN = "qwertyuiop[]asdfghjkl;'zxcvbnm,.`"
_RU = "йцукенгшщзхъфывапролджэячсмитьбюё"
_LAYOUT = str.maketrans(_EN, _RU)
_LATIN_LAYOUT_CHARS = frozenset(_EN)

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|[a-zа-я\[\];',.`]+(?:-[a-zа-я]+)*")
_PUNCT_RE = re.compile(r"[^\w\s\-\[\];',.`%°]+")
_CYRILLIC_RE = re.compile(r"[а-я]")
_LATIN_RE = re.compile(r"[a-z]")
_LAYOUT_PUNCT_RE = re.compile(r"(?<!\d)[\[\];',.`]|[\[\];',.`](?!\d)")

_UNITS = {
    "ноль": 0, "нуль": 0, "один": 1, "одна": 1, "одну": 1, "два": 2, "две": 2,
    "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
    "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
_NUMERAL_WORDS = {**_UNITS, **_TENS, **_HUNDREDS, "половина": 50, "половину": 50}


def basic_clean(text: str) -> str:
    """Регистр, ё, пунктуация. Знаки «[];',.`» остаются только в латинских словах:
    там это могут быть буквы, набранные не в той раскладке (``dsrk.xb`` = «выключи»)."""
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    words = []
    for word in text.split():
        if _CYRILLIC_RE.search(word) or not _LATIN_RE.search(word):
            word = _LAYOUT_PUNCT_RE.sub(" ", word).strip()
        if word:
            words.append(word)
    return " ".join(" ".join(words).split())


def fix_layout(word: str) -> str | None:
    """Переводит слово, набранное в латинской раскладке, в кириллицу."""
    if not word or not all(ch in _LATIN_LAYOUT_CHARS for ch in word):
        return None
    return word.translate(_LAYOUT).replace("ё", "е")


def _parse_numerals(words: list[str]) -> list[tuple[str, float | None, int]]:
    """Склеивает подряд идущие числительные: «двадцать два» → 22.

    Возвращает список (текст, значение, число исходных слов).
    """
    out: list[tuple[str, float | None, int]] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w not in _NUMERAL_WORDS:
            out.append((w, None, 1))
            i += 1
            continue
        total, used, last_rank = 0, 0, 1000
        while i + used < len(words) and words[i + used] in _NUMERAL_WORDS:
            word = words[i + used]
            rank = 100 if word in _HUNDREDS else 10 if word in _TENS else 1
            # «сто двадцать два» — разряды строго убывают; «два три» — два числа
            if rank >= last_rank or (rank == 1 and last_rank == 1):
                break
            total += _NUMERAL_WORDS[word]
            last_rank = rank if word not in ("половина", "половину") else 0
            used += 1
        out.append((str(total), float(total), used))
        i += used
    return out


class Normalizer:
    def __init__(
        self,
        stemmer: Stemmer = light_stem,
        known_word: Callable[[str], bool] | None = None,
        fix_keyboard_layout: bool = True,
    ) -> None:
        self.stemmer = stemmer
        self.known_word = known_word or (lambda _w: False)
        self.fix_keyboard_layout = fix_keyboard_layout

    def _fix_numeral(
        self, word: str, raw: str, fixed: bool, trace: list[str]
    ) -> tuple[str, str, bool]:
        """«двпдцать» → «двадцать»: опечатки в числительных (словарь маленький, DP дешёв)."""
        if len(word) < 5 or word in _NUMERAL_WORDS or self.known_word(word) or not word.isalpha():
            return word, raw, fixed
        best, best_cost = None, 1.0
        for numeral in _NUMERAL_WORDS:
            cost = weighted_distance(word, numeral, best_cost)
            if cost < best_cost or (cost == best_cost and best is None and cost <= 1.0):
                best, best_cost = numeral, cost
        if best is None:
            return word, raw, fixed
        trace.append(f"числительное: {word!r} → {best!r} (d={best_cost:.1f})")
        return best, raw, fixed

    def normalize(self, text: str) -> NormalizedQuery:
        trace: list[str] = []
        cleaned = basic_clean(text)
        if cleaned != text:
            trace.append(f"очистка: {text!r} → {cleaned!r}")
        raw_words = _TOKEN_RE.findall(cleaned)

        words: list[tuple[str, str, bool]] = []  # (нормализованное, исходное, раскладка)
        for raw in raw_words:
            word = raw
            fixed = False
            if self.fix_keyboard_layout and not self.known_word(raw):
                candidate = fix_layout(raw)
                if candidate and self.known_word(candidate):
                    trace.append(f"раскладка: {raw!r} → {candidate!r}")
                    word, fixed = candidate, True
            if not fixed:
                word = word.strip("[];',.`")
            if word:
                words.append((word, raw, fixed))

        words = [self._fix_numeral(w, raw, fixed, trace) for w, raw, fixed in words]
        tokens: list[Token] = []
        pos = 0
        for text_part, value, used in _parse_numerals([w for w, _, _ in words]):
            originals = " ".join(raw for _, raw, _ in words[pos : pos + used])
            layout = any(f for _, _, f in words[pos : pos + used])
            pos += used
            if value is not None or re.fullmatch(r"\d+(?:[.,]\d+)?", text_part):
                number = value if value is not None else float(text_part.replace(",", "."))
                if value is not None:
                    trace.append(f"число: {originals!r} → {number:g}")
                tokens.append(
                    Token(f"{number:g}", f"{number:g}", TokenKind.NUMBER, originals, number)
                )
                continue
            kind = (
                TokenKind.NEGATION
                if text_part in NEGATIONS
                else TokenKind.CONJUNCTION
                if text_part in CONJUNCTIONS
                else TokenKind.QUANTIFIER
                if text_part in QUANTIFIERS
                else TokenKind.FILLER
                if text_part in FILLERS
                else TokenKind.WORD
            )
            tokens.append(
                Token(text_part, self.stemmer(text_part), kind, originals, layout_fixed=layout)
            )
        return NormalizedQuery(original=text, tokens=tokens, trace=trace)
