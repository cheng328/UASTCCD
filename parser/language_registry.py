from __future__ import annotations

import importlib
from typing import Any, Dict


LANGUAGE_MODULES = {
    "python": "tree_sitter_python",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "c++": "tree_sitter_cpp",
    "c#": "tree_sitter_c_sharp",
}

LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "jvm": "java",
    "cpp": "c++",
    "cc": "c++",
    "cxx": "c++",
    "cs": "c#",
    "csharp": "c#",
}

_PARSER_CACHE: Dict[str, Any] = {}


def normalize_language(lang: str) -> str:
    if not lang:
        return ""
    lang = lang.strip().lower()
    return LANGUAGE_ALIASES.get(lang, lang)


def get_parser(lang: str) -> Any:
    lang = normalize_language(lang)
    if lang not in LANGUAGE_MODULES:
        raise ValueError(f"Unsupported language: {lang}")
    if lang in _PARSER_CACHE:
        return _PARSER_CACHE[lang]

    try:
        from tree_sitter import Language, Parser
    except ImportError as exc:
        raise ImportError("Missing required package: tree_sitter") from exc

    module_name = LANGUAGE_MODULES[lang]
    try:
        grammar_module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing Tree-sitter grammar package for {lang}: {module_name}"
        ) from exc

    language = Language(grammar_module.language())
    try:
        parser = Parser(language)
    except TypeError:
        parser = Parser()
        parser.set_language(language)

    _PARSER_CACHE[lang] = parser
    return parser

