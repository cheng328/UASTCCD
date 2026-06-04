from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .language_registry import get_parser, normalize_language


@dataclass
class ParseOptions:
    include_text: bool = False


class TreeSitterParser:
    def __init__(self, language: str, options: Optional[ParseOptions] = None) -> None:
        self.language = normalize_language(language)
        self.options = options or ParseOptions()
        self._parser = get_parser(self.language)

    def parse_code(self, code: str) -> Dict[str, Any]:
        code_bytes = code.encode("utf-8")
        tree = self._parser.parse(code_bytes)
        root = tree.root_node
        return {
            "language": self.language,
            "root": self._node_to_dict(root, code_bytes, child_index=0),
        }

    def _node_to_dict(self, node, code_bytes: bytes, child_index: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": node.type,
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "start_point": list(node.start_point),
            "end_point": list(node.end_point),
            "child_index": child_index,
            "named": bool(node.is_named),
            "language": self.language,
        }
        if self.options.include_text:
            data["text"] = code_bytes[node.start_byte:node.end_byte].decode(
                "utf-8", errors="replace"
            )
        if node.child_count:
            data["children"] = [
                self._node_to_dict(node.child(i), code_bytes, child_index=i)
                for i in range(node.child_count)
            ]
        return data

