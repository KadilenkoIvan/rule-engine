"""Текст для командной строки: граф деревом и строки срабатываний."""

from rule_engine.application.views import GraphView, MatchView, NodeView, SetView, TokenEdge


def render(graph: GraphView) -> str:
    """Бор, под ним шаблоны `any`. Задержку и порог показывает, где они отличаются от корня."""
    lines = ["(начало)"]

    def label(edge: TokenEdge) -> str:
        notes = []
        if edge.max_gap is not None:
            notes.append(f"{edge.max_gap:g} с")
        if edge.min_score is not None:
            notes.append(f"≥ {edge.min_score:g}")
        return f"{edge.token} [{', '.join(notes)}]" if notes else edge.token

    def walk(nodes: tuple[NodeView, ...], indent: str) -> None:
        for index, node in enumerate(nodes):
            last = index == len(nodes) - 1
            mark = f"  ⇒ {node.scenario}" if node.scenario is not None else ""
            segment = " → ".join(label(edge) for edge in node.segment)
            lines.append(f"{indent}{'└─ ' if last else '├─ '}{segment}{mark}")
            walk(node.children, indent + ("   " if last else "│  "))

    def label_set(pattern: SetView) -> str:
        # одинаковые у всех шагов пороги показаны один раз, после набора
        common = len({slot.min_score for slot in pattern.slots}) == 1
        tokens = ", ".join(
            slot.token
            if common or slot.min_score is None
            else f"{slot.token} [≥ {slot.min_score:g}]"
            for slot in pattern.slots
        )
        notes = []
        if len(pattern.slots) > 1 and pattern.max_gap is not None:
            notes.append(f"{pattern.max_gap:g} с")
        if common and pattern.slots[0].min_score is not None:
            notes.append(f"≥ {pattern.slots[0].min_score:g}")
        return f"{{{tokens}}} [{', '.join(notes)}]" if notes else f"{{{tokens}}}"

    walk(graph.roots, "")
    if graph.sets:
        lines.append("(в любом порядке)")
        for index, pattern in enumerate(graph.sets):
            branch = "└─ " if index == len(graph.sets) - 1 else "├─ "
            lines.append(f"{branch}{label_set(pattern)}  ⇒ {pattern.scenario}")
    return "\n".join(lines)


def format_match(match: MatchView) -> str:
    """`✓ взлом, уверенность 0.85 — [<дверь> 0.90, 0.0 с] [<громкий звук> 0.80, 9.0 с]`:
    у шага — его токен, уверенность и время."""
    steps = " ".join(f"[{step.token} {step.score:.2f}, {step.time} с]" for step in match.steps)
    return f"✓ {match.scenario}, уверенность {match.confidence:.2f} — {steps}"
