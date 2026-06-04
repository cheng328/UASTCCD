from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .tree_schema import ASTNode, iter_children


ROLE_LABELS = {"COND", "THEN", "ELSE", "INIT", "STEP", "BODY", "EXPR", "ARGS", "CALLEE", "VALUE"}


def _node_type(node: ASTNode) -> str:
    return str(node.get("type", "")).lower()


def _tag(node: ASTNode) -> str:
    return str(node.get("tag", "STRUCT_UNK"))


def _children(node: ASTNode) -> List[ASTNode]:
    return list(iter_children(node))


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _is_selection(node: ASTNode) -> bool:
    tag = _tag(node)
    typ = _node_type(node)
    return tag in {"COND", "SEL"} or _contains_any(
        typ,
        ("if_statement", "if", "switch_statement", "switch", "conditional_expression", "ternary"),
    )


def _is_loop(node: ASTNode) -> bool:
    tag = _tag(node)
    typ = _node_type(node)
    return tag == "LOOP" or _contains_any(
        typ,
        ("for_statement", "while_statement", "do_statement", "for_in", "foreach", "loop"),
    )


def _is_call(node: ASTNode) -> bool:
    tag = _tag(node)
    typ = _node_type(node)
    if "argument" in typ:
        return False
    return tag == "CALL" or _contains_any(typ, ("call", "invocation", "object_creation"))


def _is_argument_list(node: ASTNode) -> bool:
    typ = _node_type(node)
    return "argument_list" in typ or typ in {"arguments", "argument"}


def _is_block(node: ASTNode) -> bool:
    tag = _tag(node)
    typ = _node_type(node)
    return tag == "BLOCK" or _contains_any(typ, ("block", "body", "compound_statement", "suite"))


def _is_condition_like(node: ASTNode) -> bool:
    typ = _node_type(node)
    tag = _tag(node)
    return tag in {"COND", "EXPR"} or _contains_any(
        typ,
        ("condition", "parenthesized_expression", "comparison", "binary_expression", "boolean"),
    )


def _is_else_like(node: ASTNode) -> bool:
    typ = _node_type(node)
    return "else" in typ or "alternative" in typ


def _is_body_like(node: ASTNode) -> bool:
    return _is_block(node) or _contains_any(_node_type(node), ("body", "consequence", "then"))


def _join(parts: Iterable[str]) -> str:
    return ", ".join(part for part in parts if part)


def _role(name: str, nodes: Sequence[ASTNode]) -> str:
    if not nodes:
        return f"{name}:EMPTY"
    return f"{name}:{_join(serialize_suast(node) for node in nodes)}"


def _bucket_selection(children: Sequence[ASTNode]) -> Tuple[List[ASTNode], List[ASTNode], List[ASTNode]]:
    cond: List[ASTNode] = []
    then: List[ASTNode] = []
    else_: List[ASTNode] = []

    for idx, child in enumerate(children):
        if _is_else_like(child):
            else_.append(child)
            continue
        if not cond and (idx == 0 or _is_condition_like(child)):
            cond.append(child)
            continue
        if else_:
            else_.append(child)
        else:
            then.append(child)
    return cond, then, else_


def _bucket_loop(children: Sequence[ASTNode]) -> Tuple[List[ASTNode], List[ASTNode], List[ASTNode], List[ASTNode]]:
    init: List[ASTNode] = []
    cond: List[ASTNode] = []
    step: List[ASTNode] = []
    body: List[ASTNode] = []

    for child in children:
        typ = _node_type(child)
        if _contains_any(typ, ("initializer", "init")):
            init.append(child)
        elif _contains_any(typ, ("condition",)):
            cond.append(child)
        elif _contains_any(typ, ("update", "increment", "step")):
            step.append(child)
        elif _is_body_like(child):
            body.append(child)
        elif not cond and _is_condition_like(child):
            cond.append(child)
        elif not body:
            body.append(child)
        else:
            body.append(child)
    return init, cond, step, body


def _serialize_selection(node: ASTNode, children: Sequence[ASTNode]) -> str:
    cond, then, else_ = _bucket_selection(children)
    roles = [_role("COND", cond), _role("THEN", then)]
    if else_:
        roles.append(_role("ELSE", else_))
    return "SEL{" + "; ".join(roles) + "}"


def _serialize_loop(node: ASTNode, children: Sequence[ASTNode]) -> str:
    init, cond, step, body = _bucket_loop(children)
    roles = []
    if init:
        roles.append(_role("INIT", init))
    roles.append(_role("COND", cond))
    if step:
        roles.append(_role("STEP", step))
    roles.append(_role("BODY", body))
    return "LOOP{" + "; ".join(roles) + "}"


def _serialize_call(node: ASTNode, children: Sequence[ASTNode]) -> str:
    if not children:
        return _tag(node)
    callee = children[:1]
    args: List[ASTNode] = []
    for child in children[1:]:
        if _is_argument_list(child):
            nested = _children(child)
            args.extend(nested if nested else [child])
        else:
            args.append(child)
    return "CALL{" + "; ".join([_role("CALLEE", callee), _role("ARGS", args)]) + "}"


def _serialize_block(node: ASTNode, children: Sequence[ASTNode]) -> str:
    if not children:
        return "BLOCK"
    return "BLOCK{" + _role("BODY", children) + "}"


def serialize_suast(node: ASTNode) -> str:
    tag = _tag(node)
    children = _children(node)
    if not children:
        return tag

    if _is_selection(node):
        return _serialize_selection(node, children)
    if _is_loop(node):
        return _serialize_loop(node, children)
    if _is_call(node):
        return _serialize_call(node, children)
    if _is_block(node):
        return _serialize_block(node, children)

    inner = ", ".join(serialize_suast(child) for child in children)
    return f"{tag}({inner})"


def serialize_root(mapped_root: Dict[str, object]) -> str:
    if not isinstance(mapped_root, dict):
        return ""
    return serialize_suast(mapped_root)

