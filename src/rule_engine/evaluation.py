"""Оценка: качество поиска, безопасность, задержка, размер индекса.

Метрики качества (по запросам с эталонной командой):
  * P@1, Recall@3, Recall@5, MRR — место эталона в ранжированном списке гипотез;
  * accuracy решения — совпал ли статус (OK / CLARIFY / CONFIRM / REJECT / NOT_FOUND).

Метрики безопасности:
  * silent error — статус OK, но команда неверная (или исполнять не следовало);
  * dangerous silent — статус OK у команды с высоким риском (цель: 0);
  * over-clarify — ожидался OK, а движок переспросил (цена безопасности).

Производительность: задержка понимания запроса (p50/p95/p99), размер индекса
в памяти, время сборки. Абляции отключают уровни каскада, бэкенды нечёткого
поиска и правила политики, чтобы увидеть вклад каждого компонента.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rule_engine.engine import DATA_DIR, Engine
from rule_engine.index.lexicon import MatchType
from rule_engine.policy import PolicyConfig, Status
from rule_engine.resolver import Interpretation
from rule_engine.rules import RuleCompiler
from rule_engine.spec import Spec
from rule_engine.text.distance import _ROWS_RU
from rule_engine.text.stem import make_stemmer

DEFAULT_DATASET = DATA_DIR / "queries.jsonl"


# --------------------------------------------------------------------- dataset
@dataclass
class Example:
    query: str
    category: str
    expect: str  # ожидаемый статус; несколько допустимых — через «|»
    action: str | None = None
    targets: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    scenario: str | None = None
    value: float | None = None
    room: str | None = None  # комната-контекст
    note: str = ""

    @property
    def allowed(self) -> set[str]:
        return set(self.expect.split("|"))

    @property
    def has_gold(self) -> bool:
        return bool(self.action or self.scenario)

    def matches(self, interp: Interpretation) -> bool:
        if len(interp.commands) != 1 and not self.scenario:
            return False
        command = interp.commands[0]
        if self.scenario:
            return command.scenario == self.scenario
        return (
            command.action == self.action
            and set(command.targets) == set(self.targets)
            and (not self.objects or tuple(command.objects) == self.objects)
            and (self.value is None or command.value == self.value)
        )

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Example:
        return cls(
            query=raw["q"],
            category=raw.get("cat", "misc"),
            expect=raw.get("expect", "OK"),
            action=raw.get("action"),
            targets=tuple(raw.get("targets", ())),
            objects=tuple(raw.get("objects", ())),
            scenario=raw.get("scenario"),
            value=raw.get("value"),
            room=raw.get("room"),
            note=raw.get("note", ""),
        )


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[Example]:
    examples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            examples.append(Example.from_json(json.loads(line)))
    return examples


# ---------------------------------------------------------------------- typos
_NEIGHBOURS: dict[str, list[str]] = defaultdict(list)
for _r, _row in enumerate(_ROWS_RU):
    for _c, _ch in enumerate(_row):
        for _rr, _cc in ((_r, _c - 1), (_r, _c + 1), (_r - 1, _c), (_r + 1, _c)):
            if 0 <= _rr < len(_ROWS_RU) and 0 <= _cc < len(_ROWS_RU[_rr]):
                _NEIGHBOURS[_ch].append(_ROWS_RU[_rr][_cc])


def make_typo(word: str, rng: random.Random) -> str:
    """Одна реалистичная опечатка: соседняя клавиша, пропуск, перестановка, удвоение."""
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)  # первую букву обычно не путают
    kind = rng.choice(("neighbour", "delete", "transpose", "double"))
    if kind == "neighbour" and _NEIGHBOURS.get(word[i]):
        return word[:i] + rng.choice(_NEIGHBOURS[word[i]]) + word[i + 1 :]
    if kind == "delete":
        return word[:i] + word[i + 1 :]
    if kind == "transpose" and i + 1 < len(word):
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    return word[:i] + word[i] + word[i:]


def typo_variants(
    examples: list[Example],
    per_example: int = 1,
    seed: int = 13,
    known_word: Callable[[str], bool] | None = None,
) -> list[Example]:
    """Синтетические опечатки к однозначным запросам с эталоном.

    Для опасных команд допустимы только CONFIRM/CLARIFY: опечатка не должна
    приводить к молчаливому выполнению. Опечатки, дающие другое словарное слово
    («выключи» → «включи»), пропускаются: это уже другая корректная команда,
    и лексический слой не может её отличить (real-word error).
    """
    rng = random.Random(seed)
    out = []
    for ex in examples:
        if not ex.has_gold or ex.allowed - {"OK", "CONFIRM"} or ex.category in ("typo", "layout"):
            continue
        words = ex.query.split()
        candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
        if not candidates:
            continue
        for _ in range(per_example):
            idx = rng.choice(candidates)
            typo = make_typo(words[idx], rng)
            if typo == words[idx] or (known_word is not None and known_word(typo)):
                continue
            new_words = list(words)
            new_words[idx] = typo
            expect = "CONFIRM|CLARIFY" if "CONFIRM" in ex.allowed else "OK|CLARIFY"
            out.append(
                Example(" ".join(new_words), "typo_gen", expect, ex.action, ex.targets, ex.objects,
                        ex.scenario, ex.value, ex.room, f"из «{ex.query}»")
            )
    return out


# -------------------------------------------------------------------- metrics
@dataclass
class Outcome:
    example: Example
    status: str
    rank: int | None  # 1-based место эталона среди гипотез
    top_correct: bool
    risk: str
    latency_ms: float
    fired: list[str] = field(default_factory=list)

    @property
    def status_ok(self) -> bool:
        return self.status in self.example.allowed

    @property
    def silent_error(self) -> bool:
        if self.status != Status.OK.value:
            return False
        return "OK" not in self.example.allowed or (self.example.has_gold and not self.top_correct)

    @property
    def dangerous_silent(self) -> bool:
        return self.status == Status.OK.value and self.risk == "high"


@dataclass
class Report:
    name: str
    outcomes: list[Outcome]
    index_bytes: int = 0
    build_ms: float = 0.0

    def metrics(self, outcomes: list[Outcome] | None = None) -> dict[str, float]:
        outs = self.outcomes if outcomes is None else outcomes
        gold = [o for o in outs if o.example.has_gold]
        latencies = sorted(o.latency_ms for o in outs) or [0.0]

        def pct(p: float) -> float:
            return latencies[min(len(latencies) - 1, int(p * len(latencies)))]

        expected_ok = [o for o in outs if o.example.allowed == {"OK"}]
        return {
            "n": len(outs),
            "p_at_1": _mean(o.rank == 1 for o in gold),
            "recall_at_3": _mean(o.rank is not None and o.rank <= 3 for o in gold),
            "recall_at_5": _mean(o.rank is not None and o.rank <= 5 for o in gold),
            "mrr": _mean((1 / o.rank) if o.rank else 0.0 for o in gold),
            "decision_acc": _mean(o.status_ok for o in outs),
            "silent_error": _mean(o.silent_error for o in outs),
            "dangerous_silent": sum(o.dangerous_silent for o in outs),
            "over_clarify": _mean(o.status in ("CLARIFY", "CONFIRM") for o in expected_ok),
            "p50_ms": pct(0.5),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
        }

    def by_category(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[Outcome]] = defaultdict(list)
        for o in self.outcomes:
            groups[o.example.category].append(o)
        return {cat: self.metrics(outs) for cat, outs in sorted(groups.items())}

    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.status_ok or (o.example.has_gold and o.status == "OK" and not o.top_correct)]


def _mean(values: Iterable[float | bool]) -> float:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


def evaluate(engine: Engine, examples: list[Example], name: str = "engine", repeats: int = 1) -> Report:
    outcomes = []
    for ex in examples:
        best = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = engine.understand(ex.query, ex.room)
            elapsed = (time.perf_counter() - t0) * 1000
            best = elapsed if best is None else min(best, elapsed)
        assert best is not None
        decision = result.decision
        ranking = result.interpretations
        if decision.status is Status.CLARIFY and decision.options:
            # при уточнении пользователь видит варианты в порядке политики
            ranking = decision.options + [i for i in ranking if i not in decision.options]
        rank = next((i + 1 for i, interp in enumerate(ranking) if ex.matches(interp)), None)
        top = decision.chosen or (ranking[0] if ranking else None)
        outcomes.append(
            Outcome(
                ex, decision.status.value, rank, bool(top and ex.matches(top)),
                top.risk if top else "low", best, decision.fired,
            )
        )
    return Report(name, outcomes)


# ---------------------------------------------------------------- deep sizeof
def deep_size(obj: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_size(x, seen) for x in obj)
    elif isinstance(obj, (str, bytes, int, float, bool, type(None))) or callable(obj) and not hasattr(obj, "__dict__") and not hasattr(obj, "__slots__"):
        return size
    else:
        if hasattr(obj, "__dict__"):
            size += deep_size(vars(obj), seen)
        for slot in getattr(type(obj), "__slots__", ()):
            if hasattr(obj, slot):
                size += deep_size(getattr(obj, slot), seen)
    return size


def index_size(engine: Engine) -> int:
    lex = engine.lexicon
    parts = (lex._by_alias, lex._by_first, lex._words, lex._stem_words, lex._fuzzy_keys, lex._fuzzy, lex.entries)
    return deep_size(parts)


# ------------------------------------------------------------------ ablations
CASCADES: dict[str, frozenset[MatchType]] = {
    "exact": frozenset({MatchType.EXACT, MatchType.LAYOUT}),
    "+stem": frozenset({MatchType.EXACT, MatchType.LAYOUT, MatchType.STEM}),
    "+prefix": frozenset({MatchType.EXACT, MatchType.LAYOUT, MatchType.STEM, MatchType.PREFIX}),
    "full": frozenset(MatchType),
}
POLICY_ABLATIONS: dict[str, dict[str, Any]] = {
    "no-antonym-rule": {"rules_enabled": {"P5-antonym": False}},
    "no-risk-confirm": {"confirm_high_risk": False, "fuzzy_max_risk": "high"},
    "no-margin-rule": {"rules_enabled": {"P7-ambiguous": False}},
    "no-safety": {
        "rules_enabled": {"P5-antonym": False, "P7-ambiguous": False, "P8-low-confidence": False},
        "confirm_high_risk": False,
        "fuzzy_max_risk": "high",
    },
}


def build_engine(
    spec: Spec | None = None,
    backend: str = "trie",
    cascade: str = "full",
    policy_overrides: dict[str, Any] | None = None,
    stemmer: str = "light",
) -> tuple[Engine, float]:
    spec = spec or Spec.load(DATA_DIR / "spec.json")
    t0 = time.perf_counter()
    ruleset = RuleCompiler(make_stemmer(stemmer), backend, CASCADES[cascade]).compile(spec)
    build_ms = (time.perf_counter() - t0) * 1000
    policy = PolicyConfig.from_dict({**spec.policy, **(policy_overrides or {})})
    return Engine(ruleset, policy=policy), build_ms


def run_ablations(examples: list[Example], repeats: int = 3) -> list[Report]:
    configs: list[tuple[str, dict[str, Any]]] = [(f"cascade:{c}", {"cascade": c}) for c in CASCADES]
    configs += [(f"backend:{b}", {"backend": b}) for b in ("trie", "symspell", "ngram", "brute")]
    configs += [(f"policy:{p}", {"policy_overrides": o}) for p, o in POLICY_ABLATIONS.items()]
    reports = []
    for name, kwargs in configs:
        engine, build_ms = build_engine(**kwargs)
        report = evaluate(engine, examples, name, repeats)
        report.index_bytes = index_size(engine)
        report.build_ms = build_ms
        reports.append(report)
    return reports


# ----------------------------------------------------------------- scaling
_SYLLABLES = (
    "ка ло ми ра те до ну зи бо ве ла ре со ти мо ку на лу ди пе ро са ги фа то жи ле ви бу ца"
).split()


def synthetic_spec(n_devices: int, seed: int = 7) -> tuple[Spec, list[tuple[str, str]]]:
    """Базовый словарь + n устройств со случайными именами (для масштабирования)."""
    rng = random.Random(seed)
    data = json.loads((DATA_DIR / "spec.json").read_text(encoding="utf-8"))
    rooms = [r["id"] for r in data["rooms"]]
    names: set[str] = set()
    pairs = []
    for i in range(n_devices):
        while True:
            name = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(3, 4)))
            if name not in names:
                names.add(name)
                break
        device_id = f"syn_{i}"
        data["devices"].append(
            {"id": device_id, "type": rng.choice(["light", "socket"]), "room": rng.choice(rooms),
             "name": name, "aliases": [name]}
        )
        pairs.append((device_id, name))
    return Spec.from_dict(data), pairs


def scaling_benchmark(
    sizes: Iterable[int] = (100, 1000, 10000),
    backends: Iterable[str] = ("trie", "symspell", "ngram", "brute"),
    queries: int = 200,
    seed: int = 3,
) -> list[dict[str, Any]]:
    rows = []
    for size in sizes:
        spec, pairs = synthetic_spec(size)
        rng = random.Random(seed)
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(queries)]
        test = []
        for device_id, name in sample:
            noisy = make_typo(name, rng) if rng.random() < 0.5 else name
            test.append(Example(f"включи {noisy}", "synthetic", "OK|CLARIFY|CONFIRM", "turn_on", (device_id,)))
        for backend in backends:
            engine, build_ms = build_engine(spec, backend)
            report = evaluate(engine, test, f"{backend}@{size}")
            m = report.metrics()
            rows.append(
                {
                    "devices": size, "backend": backend, "build_ms": round(build_ms, 1),
                    "index_kb": round(index_size(engine) / 1024, 1),
                    "p_at_1": round(m["p_at_1"], 3), "recall_at_3": round(m["recall_at_3"], 3),
                    "p50_ms": round(m["p50_ms"], 3), "p95_ms": round(m["p95_ms"], 3), "p99_ms": round(m["p99_ms"], 3),
                }
            )
    return rows


# ---------------------------------------------------------------- regression
def snapshot(engine: Engine, examples: list[Example]) -> dict[str, tuple]:
    out = {}
    for ex in examples:
        result = engine.understand(ex.query, ex.room)
        top = result.decision.top
        out[ex.query] = (result.decision.status.value, top.key() if top else None)
    return out


def regression_diff(
    engine: Engine, examples: list[Example], change: Callable[[Engine], Any]
) -> tuple[Any, list[tuple[str, tuple, tuple]]]:
    """Применить изменение словаря и вернуть запросы, чей результат изменился."""
    before = snapshot(engine, examples)
    result = change(engine)
    after = snapshot(engine, examples)
    diff = [(q, before[q], after[q]) for q in before if before[q] != after[q]]
    return result, diff


# -------------------------------------------------------------------- report
def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body])


METRIC_COLUMNS = [
    "n", "p_at_1", "recall_at_3", "mrr", "decision_acc", "silent_error", "dangerous_silent",
    "over_clarify", "p50_ms", "p95_ms", "p99_ms",
]


def markdown_report(
    main: Report,
    ablations: list[Report] | None = None,
    scaling: list[dict[str, Any]] | None = None,
    stress: list[dict[str, Any]] | None = None,
) -> str:
    lines = [f"# Отчёт об оценке: {main.name}", ""]
    lines += ["## Итог", "", format_table([{"config": main.name, **main.metrics()}], ["config", *METRIC_COLUMNS]), ""]
    lines += [
        "## По категориям", "",
        format_table([{"category": c, **m} for c, m in main.by_category().items()], ["category", *METRIC_COLUMNS]),
        "",
    ]
    if ablations:
        rows = [
            {"config": r.name, **r.metrics(), "index_kb": round(r.index_bytes / 1024, 1), "build_ms": round(r.build_ms, 1)}
            for r in ablations
        ]
        lines += ["## Абляции", "", format_table(rows, ["config", *METRIC_COLUMNS, "index_kb", "build_ms"]), ""]
    if scaling:
        cols = ["devices", "backend", "build_ms", "index_kb", "p_at_1", "recall_at_3", "p50_ms", "p95_ms", "p99_ms"]
        lines += ["## Масштабирование (синтетические словари)", "", format_table(scaling, cols), ""]
    if stress:
        lines += [
            "## Стресс-тест антонимов (однобуквенные опечатки глаголов)", "",
            "antonym_ok — молчаливое выполнение противоположной команды (худшая ошибка).", "",
            format_table(stress, ["antonym_margin", "n", "correct_ok", "antonym_ok", "asked", "other"]), "",
        ]
    failures = main.failures()
    if failures:
        lines += ["## Ошибки", ""]
        for o in failures[:60]:
            ex = o.example
            gold = ex.scenario or f"{ex.action}{list(ex.targets)}"
            lines.append(
                f"- `{ex.query}` [{ex.category}] ожидалось {ex.expect} ({gold}), получено {o.status}, "
                f"ранг эталона {o.rank}, правила {o.fired}"
            )
    return "\n".join(lines) + "\n"


def summary_line(report: Report) -> str:
    m = report.metrics()
    return (
        f"{report.name}: n={m['n']} P@1={m['p_at_1']:.3f} R@3={m['recall_at_3']:.3f} MRR={m['mrr']:.3f} "
        f"acc={m['decision_acc']:.3f} silent={m['silent_error']:.3f} danger={m['dangerous_silent']} "
        f"over-clarify={m['over_clarify']:.3f} p50={m['p50_ms']:.3f}ms p95={m['p95_ms']:.3f}ms"
    )


# ------------------------------------------------------------ antonym stress
_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
_STRESS_TARGETS = {
    "turn_on": "свет в ванной", "turn_off": "свет в ванной",
    "open": "шторы в спальне", "close": "шторы в спальне",
    "start": "пылесос", "stop": "пылесос",
    "grasp": "красную чашку", "release": "красную чашку",
}


def single_edits(word: str) -> set[str]:
    out = set()
    for i in range(len(word) + 1):
        if i < len(word):
            out.add(word[:i] + word[i + 1 :])
            if i + 1 < len(word):
                out.add(word[:i] + word[i + 1] + word[i] + word[i + 2 :])
        for ch in _ALPHABET:
            out.add(word[:i] + ch + word[i:])
            if i < len(word):
                out.add(word[:i] + ch + word[i + 1 :])
    out.discard(word)
    return out


def antonym_stress(
    margins: Iterable[float] = (0.0, 0.5, 0.6, 0.7, 1.0, 1.5), per_verb: int = 40, seed: int = 11
) -> list[dict[str, Any]]:
    """Все однобуквенные опечатки глаголов, у которых есть антоним.

    Для каждого порога ``antonym_margin`` считаем: верное выполнение, молчаливое
    выполнение *антонима* (худшая ошибка), уточнение/подтверждение, прочее.
    margin = 0 — правило антонимов выключено.
    """
    base, _ = build_engine()
    rng = random.Random(seed)
    cases: list[tuple[str, str, str]] = []
    for action in base.spec.actions.values():
        if not action.antonym or action.id not in _STRESS_TARGETS:
            continue
        for alias in action.aliases:
            if " " in alias.alias or len(alias.alias) < 5:
                continue
            variants = sorted(v for v in single_edits(alias.alias) if not base.lexicon.known_word(v))
            for v in rng.sample(variants, min(per_verb, len(variants))):
                cases.append((f"{v} {_STRESS_TARGETS[action.id]}", action.id, action.antonym))
    rows = []
    for margin in margins:
        overrides: dict[str, Any] = {"antonym_margin": margin}
        if margin == 0:
            overrides["rules_enabled"] = {"P5-antonym": False}
        engine, _ = build_engine(policy_overrides=overrides)
        counts = defaultdict(int)
        for query, gold, antonym in cases:
            decision = engine.understand(query).decision
            action = decision.chosen.commands[0].action if decision.chosen and decision.chosen.commands else None
            if decision.status is Status.OK and action == gold:
                counts["correct_ok"] += 1
            elif decision.status is Status.OK and action == antonym:
                counts["antonym_ok"] += 1
            elif decision.status in (Status.CLARIFY, Status.CONFIRM):
                counts["asked"] += 1
            else:
                counts["other"] += 1
        n = len(cases)
        rows.append(
            {"antonym_margin": margin, "n": n, **{k: round(counts[k] / n, 4) for k in ("correct_ok", "antonym_ok", "asked", "other")}}
        )
    return rows
