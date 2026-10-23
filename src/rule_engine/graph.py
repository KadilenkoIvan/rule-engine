"""Префиксный граф сценариев: правила с общим началом делят вершины."""

from dataclasses import dataclass, field

from rule_engine.rules import RuleSet


@dataclass(eq=False)
class Node:
    """Сравнивается по identity: вершина служит ключом для активных совпадений."""

    children: dict[str, "Node"] = field(default_factory=dict)
    scenario: str | None = None  # сценарий, один из шаблонов которого завершается в этой вершине


def build_graph(rules: RuleSet) -> Node:
    root = Node()
    for scenario in rules.scenarios:
        for pattern in scenario.patterns:
            node = root
            for token in pattern:
                if token not in node.children:
                    node.children[token] = Node()
                node = node.children[token]
            if node.scenario is not None:
                raise ValueError(
                    f"сценарии {node.scenario!r} и {scenario.name!r} задают одну и ту же "
                    f"последовательность токенов {list(pattern)}"
                )
            node.scenario = scenario.name
    return root


def graph_size(root: Node) -> int:
    """Число вершин без корня."""
    return sum(1 + graph_size(child) for child in root.children.values())


def render(root: Node) -> str:
    lines = ["(начало)"]

    def walk(node: Node, indent: str) -> None:
        items = list(node.children.items())
        for index, (token, child) in enumerate(items):
            last = index == len(items) - 1
            mark = f"  ⇒ {child.scenario}" if child.scenario is not None else ""
            lines.append(f"{indent}{'└─ ' if last else '├─ '}{token}{mark}")
            walk(child, indent + ("   " if last else "│  "))

    walk(root, "")
    return "\n".join(lines)
