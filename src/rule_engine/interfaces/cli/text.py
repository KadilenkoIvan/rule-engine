"""Text for the command line: the graph as a tree and match lines."""

from rule_engine.application.tokens import token_label
from rule_engine.application.views import (
    GraphView,
    MatchView,
    NodeView,
    SetView,
    SilenceEdge,
    TokenEdge,
)


def render(graph: GraphView) -> str:
    """The trie as a tree, then the `any` patterns.

    Gaps and thresholds are shown where they differ from the root rules.
    """
    lines = ["(начало)"]
    _walk(graph.roots, "", lines)
    if graph.sets:
        lines.append("(в любом порядке)")
        for index, pattern in enumerate(graph.sets):
            branch = "└─ " if index == len(graph.sets) - 1 else "├─ "
            lines.append(f"{branch}{_label_set(pattern)}  ⇒ {pattern.scenario}")
    return "\n".join(lines)


def _label(edge: TokenEdge | SilenceEdge) -> str:
    """Edge label: the token or the silence, with its own gap and threshold."""
    if isinstance(edge, SilenceEdge):
        before = f", назад {edge.before:g} с" if edge.before else ""
        tokens = ", ".join(map(token_label, edge.tokens))
        text = f"тишина {edge.duration:g} с{before}: {tokens}"
        return text if edge.min_score is None else f"{text} [≥ {edge.min_score:g}]"
    notes = []
    if edge.max_gap is not None:
        notes.append(f"{edge.max_gap:g} с")
    if edge.min_score is not None:
        notes.append(f"≥ {edge.min_score:g}")
    token = token_label(edge.token)
    return f"{token} [{', '.join(notes)}]" if notes else token


def _walk(nodes: tuple[NodeView, ...], indent: str, lines: list[str]) -> None:
    """Appends the tree lines of `nodes` and their subtrees to `lines`."""
    for index, node in enumerate(nodes):
        last = index == len(nodes) - 1
        mark = f"  ⇒ {node.scenario}" if node.scenario is not None else ""
        lines.append(f"{indent}{'└─ ' if last else '├─ '}{_label(node.edge)}{mark}")
        _walk(node.children, indent + ("   " if last else "│  "), lines)


def _label_set(pattern: SetView) -> str:
    """Label of an `any` pattern: its steps, the gap and the thresholds."""
    # одинаковые у всех шагов пороги показаны один раз, после набора
    common = len({slot.min_score for slot in pattern.slots}) == 1
    tokens = ", ".join(
        token_label(slot.token)
        if common or slot.min_score is None
        else f"{token_label(slot.token)} [≥ {slot.min_score:g}]"
        for slot in pattern.slots
    )
    notes = []
    if len(pattern.slots) > 1 and pattern.max_gap is not None:
        notes.append(f"{pattern.max_gap:g} с")
    if common and pattern.slots[0].min_score is not None:
        notes.append(f"≥ {pattern.slots[0].min_score:g}")
    return f"{{{tokens}}} [{', '.join(notes)}]" if notes else f"{{{tokens}}}"


def format_match(match: MatchView, silence: float | None = None) -> str:
    """Match line: `✓ взлом, уверенность 0.85 — [<дверь> 0.90, 0.0 с] [<громкий звук> 0.80, 9.0 с]`.

    Each step shows its token, score and time; a held match also shows its release deadline.

    Args:
        match: the match.
        silence: end of the silence window the scenario fired by, without an event.
    """
    steps = " ".join(f"[{step.token} {step.score:.2f}, {step.time} с]" for step in match.steps)
    line = f"✓ {match.scenario}, уверенность {match.confidence:.2f} — {steps}"
    if match.released is not None:
        return f"{line} — выдано в {match.released} с"
    return line if silence is None else f"{line} — тишина до {silence} с"
