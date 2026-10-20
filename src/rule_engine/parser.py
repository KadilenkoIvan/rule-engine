"""Разбор запроса на слоты: k-best сегментация динамическим программированием.

Позиции 0..n между токенами — состояния DP. Из позиции ``i`` можно:
  * взять совпадение синонима ``[i, j)`` → в ``j`` с выигрышем Σ качества слов;
  * взять служебный токен (число, отрицание, союз, квантор) как отдельный элемент;
  * пропустить токен со штрафом (неизвестное слово штрафуется сильнее служебного).

``beam[i]`` хранит K лучших частичных разборов до позиции ``i`` (Витерби с
лучом), поэтому альтернативы — «свет на кухне» как имя устройства или как
«свет» + «кухня» — доживают до этапа выбора команды, а не теряются жадно.

Затем каждый разбор раскладывается на подкоманды (clauses): новое действие
открывает новую подкоманду; часть после союза без своего действия
присоединяется к предыдущей («выключи свет и телевизор»).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rule_engine.index.lexicon import Kind, Lexicon, SpanMatch
from rule_engine.text.normalize import NormalizedQuery, Token, TokenKind

SKIP_PENALTY = {
    TokenKind.FILLER: 0.0,
    TokenKind.WORD: 0.45,  # неизвестное слово снижает уверенность
}
MATCHED_SKIP_PENALTY = 0.6  # пропуск слова, у которого было совпадение в словаре
NUMBER_GAIN = 0.3
QUANT_GAIN = 0.2


@dataclass(frozen=True, slots=True)
class Item:
    kind: str  # span | number | neg | conj | quant | skip
    start: int
    end: int
    span: SpanMatch | None = None
    value: float | None = None

    def signature(self) -> tuple:
        return (self.kind, self.start, self.end, self.span.entry.id if self.span else None)


@dataclass
class Clause:
    action: SpanMatch | None = None
    scenario: SpanMatch | None = None
    targets: list[SpanMatch] = field(default_factory=list)
    rooms: list[SpanMatch] = field(default_factory=list)
    attrs: list[SpanMatch] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    quantifier: bool = False
    negated: bool = False
    after_conjunction: bool = False

    @property
    def empty(self) -> bool:
        return not (self.action or self.scenario or self.targets or self.rooms or self.attrs or self.values)

    @property
    def has_verb(self) -> bool:
        return self.action is not None or self.scenario is not None

    def absorb(self, other: Clause) -> None:
        self.targets += other.targets
        self.rooms += other.rooms
        self.attrs += other.attrs
        self.values += other.values
        self.quantifier |= other.quantifier
        self.negated |= other.negated

    def spans(self) -> list[SpanMatch]:
        head = [s for s in (self.action, self.scenario) if s is not None]
        return head + self.targets + self.rooms + self.attrs


@dataclass
class Parse:
    score: float
    items: tuple[Item, ...]
    clauses: list[Clause]
    coverage: float  # доля содержательных токенов, покрытых словарём
    unknown: list[str]  # пропущенные содержательные слова


class Parser:
    def __init__(self, lexicon: Lexicon, beam: int = 8) -> None:
        self.lexicon = lexicon
        self.beam = beam

    def parse(self, query: NormalizedQuery) -> list[Parse]:
        tokens = query.tokens
        n = len(tokens)
        if n == 0:
            return []
        spans_at: dict[int, list[SpanMatch]] = defaultdict(list)
        for span in self.lexicon.match_spans(tokens):
            spans_at[span.start].append(span)
        matched_positions = {i for spans in spans_at.values() for s in spans for i in range(s.start, s.end)}

        beams: list[list[tuple[float, tuple[Item, ...]]]] = [[] for _ in range(n + 1)]
        beams[0] = [(0.0, ())]
        for i in range(n):
            beams[i] = self._prune(beams[i])
            token = tokens[i]
            for score, items in beams[i]:
                special = self._special(token, i)
                if special is not None:
                    gain, item = special
                    beams[i + 1].append((score + gain, items + (item,)))
                else:
                    penalty = (
                        MATCHED_SKIP_PENALTY if i in matched_positions
                        else SKIP_PENALTY.get(token.kind, 0.45)
                    )
                    beams[i + 1].append((score - penalty, items + (Item("skip", i, i + 1),)))
                for span in spans_at.get(i, ()):
                    gain = sum(w.quality for w in span.words) * span.entry.weight
                    beams[span.end].append((score + gain, items + (Item("span", i, span.end, span),)))
        parses = [self._build(score, items, tokens) for score, items in self._prune(beams[n])]
        parses.sort(key=lambda p: -p.score)
        return parses

    def _prune(
        self, beam: list[tuple[float, tuple[Item, ...]]]
    ) -> list[tuple[float, tuple[Item, ...]]]:
        best: dict[tuple, tuple[float, tuple[Item, ...]]] = {}
        for score, items in beam:
            sig = tuple(it.signature() for it in items)
            if sig not in best or score > best[sig][0]:
                best[sig] = (score, items)
        return sorted(best.values(), key=lambda x: -x[0])[: self.beam]

    @staticmethod
    def _special(token: Token, i: int) -> tuple[float, Item] | None:
        match token.kind:
            case TokenKind.NUMBER:
                return NUMBER_GAIN, Item("number", i, i + 1, value=token.value)
            case TokenKind.NEGATION:
                return 0.0, Item("neg", i, i + 1)
            case TokenKind.CONJUNCTION:
                return 0.0, Item("conj", i, i + 1)
            case TokenKind.QUANTIFIER:
                return QUANT_GAIN, Item("quant", i, i + 1)
        return None

    @staticmethod
    def _build(score: float, items: tuple[Item, ...], tokens: list[Token]) -> Parse:
        clauses = [Clause()]
        for item in items:
            current = clauses[-1]
            if item.kind == "conj":
                if not current.empty:
                    clauses.append(Clause(after_conjunction=True))
            elif item.kind == "neg":
                current.negated = True
            elif item.kind == "quant":
                current.quantifier = True
            elif item.kind == "number" and item.value is not None:
                current.values.append(item.value)
            elif item.kind == "span" and item.span is not None:
                span = item.span
                kind = span.entry.kind
                if kind in (Kind.ACTION, Kind.SCENARIO):
                    if current.has_verb:
                        current = Clause()
                        clauses.append(current)
                    if kind is Kind.ACTION:
                        current.action = span
                    else:
                        current.scenario = span
                elif kind in (Kind.DEVICE, Kind.DEVICE_TYPE, Kind.OBJECT_TYPE):
                    current.targets.append(span)
                elif kind is Kind.ROOM:
                    current.rooms.append(span)
                elif kind is Kind.ATTRIBUTE:
                    current.attrs.append(span)

        # Часть после союза без своего глагола относится к предыдущей подкоманде.
        merged: list[Clause] = []
        for clause in clauses:
            if clause.empty and not clause.negated:
                continue
            if merged and not clause.has_verb and (clause.after_conjunction or not merged[-1].has_verb):
                merged[-1].absorb(clause)
            elif merged and clause.has_verb and not merged[-1].has_verb and not merged[-1].after_conjunction:
                # «свет на кухне ... включи»: глагол в конце относится к уже названной цели
                clause.absorb(merged[-1])
                merged[-1] = clause
            else:
                merged.append(clause)

        content = [i for i, t in enumerate(tokens) if t.kind in (TokenKind.WORD, TokenKind.NUMBER)]
        covered = set()
        unknown = []
        for item in items:
            if item.kind in ("span", "number"):
                covered.update(range(item.start, item.end))
            elif item.kind == "skip" and tokens[item.start].kind is TokenKind.WORD:
                unknown.append(tokens[item.start].original)
        coverage = (len([i for i in content if i in covered]) / len(content)) if content else 0.0
        return Parse(score, items, merged or [Clause()], coverage, unknown)
