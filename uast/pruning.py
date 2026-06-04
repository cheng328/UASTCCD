from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .tree_schema import ASTNode


COMMON_DROP_TYPES = {
    "ERROR",
    "ERROR_NODE",
    "MISSING",
    "comment",
    "line_comment",
    "block_comment",
    "documentation",
    "documentation_comment",
    "doc_comment",
}

ADMIN_NODE_TYPES = {
    "package_declaration",
    "import_declaration",
    "import_statement",
    "module",
    "module_declaration",
    "using_directive",
    "preproc_include",
    "preproc_def",
    "preproc_function_def",
    "preproc_call",
    "namespace_definition",
}

CONTAINER_DROP_TYPES = {
    "module",
    "translation_unit",
    "program",
    "source_file",
    "compilation_unit",
}

PUNCTUATION_TYPES = {
    "(", ")", "{", "}", "[", "]", ";", ",", ".", ":", "::", "?", "=>",
}

KEYWORD_TOKEN_TYPES = {
    "if",
    "else",
    "switch",
    "case",
    "default",
    "for",
    "while",
    "do",
    "try",
    "catch",
    "finally",
    "class",
    "struct",
    "interface",
    "enum",
    "public",
    "private",
    "protected",
    "static",
    "final",
    "abstract",
    "new",
    "throws",
    "throw",
    "import",
    "package",
    "using",
    "namespace",
    "def",
    "lambda",
}


def is_punctuation(node_type: str) -> bool:
    if node_type in PUNCTUATION_TYPES:
        return True
    if len(node_type) == 1 and not node_type.isalnum():
        return True
    return node_type.startswith('"') and node_type.endswith('"')


def should_drop(node_type: str, granularity: str) -> bool:
    if node_type in COMMON_DROP_TYPES:
        return True
    if is_punctuation(node_type):
        return True
    if node_type in KEYWORD_TOKEN_TYPES:
        return True
    if granularity in {"function", "block"} and node_type in CONTAINER_DROP_TYPES:
        return True
    if granularity in {"function", "block"} and node_type in ADMIN_NODE_TYPES:
        return True
    return False


def prune_node_list(node: ASTNode, granularity: str) -> List[ASTNode]:
    node_type = str(node.get("type", ""))
    if should_drop(node_type, granularity):
        if node_type in CONTAINER_DROP_TYPES:
            lifted: List[ASTNode] = []
            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        lifted.extend(prune_node_list(child, granularity))
            return lifted
        return []

    children = node.get("children")
    out_node = dict(node)
    if isinstance(children, list):
        pruned_children: List[ASTNode] = []
        for child in children:
            if not isinstance(child, dict):
                continue
            pruned_children.extend(prune_node_list(child, granularity))
        if pruned_children:
            for idx, child in enumerate(pruned_children):
                child["child_index"] = idx
            out_node["children"] = pruned_children
        else:
            out_node.pop("children", None)

    return [out_node]


def prune_node(node: ASTNode, granularity: str) -> Optional[ASTNode]:
    nodes = prune_node_list(node, granularity)
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    return {"type": "ROOT", "children": nodes}


def prune_ast(ast: Dict[str, object], granularity: str = "function") -> Dict[str, object]:
    root = ast.get("root") if isinstance(ast.get("root"), dict) else ast
    if not isinstance(root, dict):
        return ast

    pruned_root = prune_node(root, granularity)
    if pruned_root is None:
        pruned_root = {"type": "EMPTY"}

    out = dict(ast)
    if "root" in out and isinstance(out.get("root"), dict):
        out["root"] = pruned_root
    else:
        out = pruned_root
    return out

