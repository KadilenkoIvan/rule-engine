"""Layout of the trie for SVG: a row per branch, columns as wide as their edge labels.

The first child stays in the parent's row, the others go below, the edge going down like `└─`.
A node where a scenario ends has its label on the right, so its children start a row lower.
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
    """Node with the edge into it."""

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
    """Size of the picture, the root and the placed nodes."""

    width: float
    height: float
    root_x: float
    root_y: float
    boxes: tuple[Box, ...]


def label(edge: TokenEdge | SilenceEdge) -> str:
    """Edge label; gap and threshold where they differ from the root."""
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
    """Places the trie nodes in rows and columns for the SVG."""
    grid = _Grid()
    row = 0
    for root in graph.roots:
        row = grid.place(root, None, 1, row)
    xs = _columns(grid)
    boxes = []
    width = xs[0] + RADIUS
    for node in grid.nodes:
        box = _box(node, grid, xs)
        boxes.append(box)
        end = box.x + RADIUS
        if node.scenario is not None:
            end += 10 + (len(node.scenario) + 4) * CHAR
        width = max(width, end)
    return Layout(width + MARGIN, _y(max(row, 1)) - ROW / 2 + MARGIN, xs[0], _y(0), tuple(boxes))


class _Grid:
    """Rows and columns (depths) of the nodes, in walk order."""

    def __init__(self) -> None:
        self.depths: dict[int, int] = {}
        self.rows: dict[int, int] = {}
        self.parents: dict[int, int | None] = {}
        self.nodes: list[NodeView] = []

    def place(self, node: NodeView, parent: int | None, depth: int, row: int) -> int:
        """Puts the node in row `row`; returns the first free row below its subtree."""
        self.nodes.append(node)
        self.depths[node.id], self.rows[node.id], self.parents[node.id] = depth, row, parent
        if not node.children:
            return row + 1
        child_row = row + 1 if node.scenario is not None else row
        for child in node.children:
            child_row = self.place(child, node.id, depth + 1, child_row)
        return child_row


def _columns(grid: _Grid) -> list[float]:
    """X of the nodes of each depth: a column is as wide as its longest edge label."""
    columns: dict[int, float] = {}
    for node in grid.nodes:
        text = len(label(node.edge)) * CHAR
        columns[grid.depths[node.id]] = max(columns.get(grid.depths[node.id], 0.0), text)
    xs = [MARGIN + RADIUS]
    for depth in range(1, max(columns, default=0) + 1):
        xs.append(xs[-1] + 2 * RADIUS + 2 * PAD + columns[depth])
    return xs


def _y(row: int) -> float:
    """Y of a row."""
    return MARGIN + 6 + row * ROW


def _box(node: NodeView, grid: _Grid, xs: list[float]) -> Box:
    """Node at its place with the edge path from its parent."""
    parent = grid.parents[node.id]
    parent_row = 0 if parent is None else grid.rows[parent]
    px, py = xs[grid.depths[node.id] - 1], _y(parent_row)
    cx, cy = xs[grid.depths[node.id]], _y(grid.rows[node.id])
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
    return Box(
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
