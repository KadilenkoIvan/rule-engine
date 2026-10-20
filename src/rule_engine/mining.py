"""Выучивание правил из потока команд: «движок принимает набор и составляет набор правил».

Вход — журнал выполненных команд (или сырые тексты, которые движок сначала
разберёт). Выход — кандидаты в сценарии и правила «после A обычно B».

1. Журнал режется на сессии (по id или по паузе между командами).
2. Частые последовательные шаблоны ищутся поуровнево (как Apriori): шаблон
   длины k+1 — продолжение частого шаблона длины k частым элементом.
3. Поддержка шаблона считается DP «вхождение подпоследовательности с
   ограничением на разрыв»: между соседними элементами шаблона в сессии не
   больше ``max_gap`` посторонних команд. Жадный поиск здесь неверен — ранний
   выбор вхождения может нарушить разрыв дальше, а DP перебирает все.
4. Оставляются *замкнутые* шаблоны (нет надшаблона с той же поддержкой).
5. Шаблоны превращаются в сценарии (``source = "mined"``) и в правила
   следования с confidence = support(p + x) / support(p).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rule_engine.rules import Rule, RuleKind
from rule_engine.spec import AliasSpec, ScenarioSpec, ScenarioStep, Spec, TargetSelector

Item = tuple[str, str]  # (действие, устройство)


def contains_with_gap(session: Sequence[Item], pattern: Sequence[Item], max_gap: int) -> bool:
    """Есть ли ``pattern`` в ``session`` как подпоследовательность с разрывами ≤ max_gap.

    reach[k][i] — можно сопоставить pattern[:k+1] так, что pattern[k] стоит на позиции i:

        reach[0][i] = session[i] == pattern[0]
        reach[k][i] = session[i] == pattern[k] and ∃ i' ∈ [i−max_gap−1, i−1]: reach[k−1][i']

    Префиксные суммы делают проверку окна O(1), итого O(len(session)·len(pattern)).
    """
    n, m = len(session), len(pattern)
    if m == 0:
        return True
    if m > n:
        return False
    prev = [session[i] == pattern[0] for i in range(n)]
    for k in range(1, m):
        prefix = [0] * (n + 1)
        for i, ok in enumerate(prev):
            prefix[i + 1] = prefix[i] + ok
        cur = [False] * n
        for i in range(n):
            if session[i] != pattern[k]:
                continue
            lo = max(0, i - max_gap - 1)
            cur[i] = prefix[i] - prefix[lo] > 0
        prev = cur
        if not any(prev):
            return False
    return any(prev)


@dataclass
class Pattern:
    items: tuple[Item, ...]
    support: int  # число сессий
    sessions: frozenset[int] = field(default_factory=frozenset)


@dataclass
class MiningResult:
    patterns: list[Pattern]
    scenarios: list[ScenarioSpec]
    rules: list[Rule]
    sessions: int
    notes: list[str]

    def to_json(self) -> dict[str, Any]:
        from rule_engine.spec import scenario_to_json

        return {
            "sessions": self.sessions,
            "scenarios": [scenario_to_json(s) for s in self.scenarios],
            "rules": [{"id": r.id, "when": r.when, "then": r.then, **dict(r.data)} for r in self.rules],
            "notes": self.notes,
        }


def sessionize(events: Iterable[dict[str, Any]], gap_ms: float = 30 * 60 * 1000) -> list[list[Item]]:
    """События ``{"session"?, "ts", "action", "device"}`` → список сессий."""
    by_session: dict[str, list[tuple[float, Item]]] = defaultdict(list)
    auto_id, last_ts = 0, None
    for event in sorted(events, key=lambda e: (str(e.get("session", "")), e.get("ts", 0))):
        if "session" in event:
            key = str(event["session"])
        else:
            ts = float(event.get("ts", 0))
            if last_ts is not None and ts - last_ts > gap_ms:
                auto_id += 1
            last_ts = ts
            key = f"auto-{auto_id}"
        by_session[key].append((float(event.get("ts", 0)), (event["action"], event["device"])))
    return [[item for _, item in sorted(evts)] for _, evts in sorted(by_session.items())]


class RuleMiner:
    def __init__(
        self,
        spec: Spec,
        min_support: float = 0.3,
        min_len: int = 2,
        max_len: int = 6,
        max_gap: int = 2,
        min_confidence: float = 0.6,
    ) -> None:
        self.spec = spec
        self.min_support = min_support
        self.min_len = min_len
        self.max_len = max_len
        self.max_gap = max_gap
        self.min_confidence = min_confidence

    def mine(self, sessions: list[list[Item]]) -> MiningResult:
        n = len(sessions)
        notes: list[str] = []
        if n == 0:
            return MiningResult([], [], [], 0, ["пустой журнал"])
        threshold = max(2, int(self.min_support * n + 0.999))
        notes.append(f"сессий: {n}, порог поддержки: {threshold}")

        counts: dict[Item, set[int]] = defaultdict(set)
        for sid, session in enumerate(sessions):
            for item in session:
                counts[item].add(sid)
        frequent_items = sorted(i for i, s in counts.items() if len(s) >= threshold)
        level = [Pattern((i,), len(counts[i]), frozenset(counts[i])) for i in frequent_items]
        all_patterns = list(level)
        while level and len(level[0].items) < self.max_len:
            next_level = []
            for pattern in level:
                for item in frequent_items:
                    if item in pattern.items:
                        continue
                    candidate = pattern.items + (item,)
                    # анти-монотонность: искать только в сессиях, где есть префикс
                    hits = frozenset(
                        sid for sid in pattern.sessions & counts[item]
                        if contains_with_gap(sessions[sid], candidate, self.max_gap)
                    )
                    if len(hits) >= threshold:
                        next_level.append(Pattern(candidate, len(hits), hits))
            all_patterns += next_level
            level = next_level

        support = {p.items: p.support for p in all_patterns}
        # Замкнутые шаблоны: нет надшаблона с не меньшей поддержкой. Сравнение «≥»,
        # а не «=»: из-за ограничения на разрыв поддержка подшаблона может быть даже меньше.
        closed = [
            p for p in all_patterns
            if len(p.items) >= self.min_len
            and not any(
                len(q.items) > len(p.items) and q.support >= p.support and _is_subsequence(p.items, q.items)
                for q in all_patterns
            )
        ]
        closed.sort(key=lambda p: (-p.support, -len(p.items), p.items))

        rules = []
        for p in all_patterns:
            if len(p.items) < 2:
                continue
            prefix, last = p.items[:-1], p.items[-1]
            confidence = p.support / support[prefix]
            if confidence >= self.min_confidence:
                when = " → ".join(f"{a}({d})" for a, d in prefix)
                rules.append(
                    Rule(
                        f"seq:{len(rules) + 1}", RuleKind.MINED,
                        f"выполнено {when}", f"предложить {last[0]}({last[1]})", "mined",
                        {"support": p.support / n, "confidence": round(confidence, 3)},
                    )
                )

        scenarios = []
        for idx, p in enumerate(closed, start=1):
            similar = self._similar_existing(p.items)
            if similar:
                notes.append(f"шаблон {_fmt(p.items)} похож на существующий сценарий «{similar}» — пропущен")
                continue
            title = "Выученный: " + ", ".join(
                f"{self.spec.actions[a].title} {self.spec.devices[d].name}" for a, d in p.items
            )
            scenarios.append(
                ScenarioSpec(
                    id=f"mined_{idx}",
                    aliases=(AliasSpec(f"привычка {idx}"),),
                    steps=tuple(ScenarioStep(a, TargetSelector(device=d)) for a, d in p.items),
                    ordered=True,
                    title=title,
                    source="mined",
                    support=p.support / n,
                )
            )
        return MiningResult(closed, scenarios, rules, n, notes)

    def _similar_existing(self, items: tuple[Item, ...], threshold: float = 0.8) -> str | None:
        mined = set(items)
        for scenario in self.spec.scenarios.values():
            expanded = set()
            for step in scenario.steps:
                sel = step.target
                devices = [sel.device] if sel.device else [d.id for d in self.spec.devices_of(sel.type or "", sel.room)]
                expanded |= {(step.action, d) for d in devices}
            if expanded and len(mined & expanded) / len(mined) >= threshold:
                return scenario.title
        return None


def _is_subsequence(short: Sequence[Item], long: Sequence[Item]) -> bool:
    it = iter(long)
    return all(any(x == y for y in it) for x in short)


def _fmt(items: Sequence[Item]) -> str:
    return " → ".join(f"{a}({d})" for a, d in items)


def load_events(path: str | Path) -> list[dict[str, Any]]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            events.append(json.loads(line))
    return events
