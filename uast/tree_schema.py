from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional


ASTNode = Dict[str, object]


def is_ast_node(node: object) -> bool:
    return isinstance(node, dict) and ("type" in node or "tag" in node)


def iter_children(node: ASTNode) -> Iterable[ASTNode]:
    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            if is_ast_node(child):
                yield child


def walk_preorder(root: ASTNode) -> Iterator[ASTNode]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        children = list(iter_children(node))
        stack.extend(reversed(children))


def get_language(ast: Dict[str, object]) -> Optional[str]:
    lang = ast.get("language")
    if isinstance(lang, str):
        return lang
    return None

