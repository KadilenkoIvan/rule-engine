"""Раскладка бора для SVG: строки как у графа в README, ширина колонки — по подписям рёбер.

Первый потомок идёт по строке родителя, остальные — ниже: ребро спускается из вершины, как
`└─`. Если в вершине кончается сценарий, его подпись стоит справа, и потомки начинаются со
следующей строки.
"""

from typing import NamedTuple

from rule_engine.application.tokens import token_label
from rule_engine.application.views import GraphView, NodeView, SilenceEdge, TokenEdge

CHAR = 7.9  # ширина знака моноширинного шрифта 13 px
ROW = 36  # высота строки
RADIUS = 6
PAD = 12  # поля подписи ребра слева и справа
MARGIN = 20


class Box(NamedTuple):
    """Вершина с ребром, которое в неё ведёт."""

    id: int
    parent: int | None  # None — ребро из корня
    x: float
    y: float
    path: str  # SVG-путь ребра от родителя
    label: str
    label_x: float  # середина подписи
    scenario: str | None
    silence: bool  # ребро-тишина


class Layout(NamedTuple):
    width: float
    height: float
    root_x: float
    root_y: float
    boxes: tuple[Box, ...]


def label(edge: TokenEdge | SilenceEdge) -> str:
    """Подпись ребра. Задержка и порог — там, где отличаются от корня."""
    if isinstance(edge, SilenceEdge):
        notes = [f"{edge.duration:g} с"]
        if edge.before:
            notes.append(f"назад {edge.before:g} с")
        if edge.min_score is not None:
            notes.append(f"≥ {edge.min_score:g}")
        return " · ".join([f"тишина: {', '.join(map(token_label, edge.tokens))}", *notes])
    notes = []
    if edge.max_gap is not None:
        notes.append(f"{edge.max_gap:g} с")
    if edge.min_score is not None:
        notes.append(f"≥ {edge.min_score:g}")
    return " · ".join([token_label(edge.token), *notes])


def layout(graph: GraphView) -> Layout:
    depths: dict[int, int] = {}
    rows: dict[int, int] = {}
    parents: dict[int, int | None] = {}
    nodes: list[NodeView] = []

    def place(node: NodeView, parent: int | None, depth: int, row: int) -> int:
        """Ставит вершину в строку `row`; возвращает первую свободную строку под поддеревом."""
        nodes.append(node)
        depths[node.id], rows[node.id], parents[node.id] = depth, row, parent
        if not node.children:
            return row + 1
        child_row = row + 1 if node.scenario is not None else row
        for child in node.children:
            child_row = place(child, node.id, depth + 1, child_row)
        return child_row

    row = 0
    for root in graph.roots:
        row = place(root, None, 1, row)

    # ширина колонки — по самой длинной подписи ребра, которое в неё ведёт
    columns: dict[int, float] = {}
    for node in nodes:
        text = len(label(node.edge)) * CHAR
        columns[depths[node.id]] = max(columns.get(depths[node.id], 0.0), text)
    xs = [MARGIN + RADIUS]
    for depth in range(1, max(columns, default=0) + 1):
        xs.append(xs[-1] + 2 * RADIUS + 2 * PAD + columns[depth])

    def y(row: int) -> float:
        return MARGIN + 6 + row * ROW

    boxes = []
    width = xs[0] + RADIUS
    for node in nodes:
        parent = parents[node.id]
        parent_row = 0 if parent is None else rows[parent]
        px, py = xs[depths[node.id] - 1], y(parent_row)
        cx, cy = xs[depths[node.id]], y(rows[node.id])
        if cy == py:
            start = px + RADIUS
            path = f"M{start:g},{py:g} H{cx - RADIUS:g}"
        else:
            # спуск из-под вершины: справа от неё может стоять подпись сценария
            start = px
            path = (
                f"M{px:g},{py + RADIUS:g} V{cy - 6:g} Q{px:g},{cy:g} {px + 6:g},{cy:g} "
                f"H{cx - RADIUS:g}"
            )
        boxes.append(
            Box(
                node.id,
                parent,
                cx,
                cy,
                path,
                label(node.edge),
                (start + cx - RADIUS) / 2,
                node.scenario,
                isinstance(node.edge, SilenceEdge),
            )
        )
        end = cx + RADIUS
        if node.scenario is not None:
            end += 10 + (len(node.scenario) + 4) * CHAR
        width = max(width, end)
    return Layout(width + MARGIN, y(max(row, 1)) - ROW / 2 + MARGIN, xs[0], y(0), tuple(boxes))
