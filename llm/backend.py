from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ChatConfig:
    endpoint: str = "http://localhost:11434/api/chat"
    model: str = "qwen3:30b"
    timeout: Optional[float] = None
    stream: bool = False
    format_json: bool = True


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text:
        return None
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    return obj
    return None


def _extract_chunk(obj: Dict[str, Any]) -> str:
    msg = obj.get("message") or {}
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str) and content:
            return content
    response = obj.get("response")
    if isinstance(response, str) and response:
        return response
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        delta = first.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
        msg = first.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content:
            return content
    return ""


def call_chat_json(messages: List[Dict[str, str]], config: ChatConfig) -> Dict[str, Any]:
    import requests

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": config.stream,
    }
    if config.format_json:
        payload["format"] = "json"

    response = requests.post(
        config.endpoint,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        stream=config.stream,
        timeout=config.timeout,
    )
    response.raise_for_status()

    if not config.stream:
        obj = response.json()
        text = _extract_chunk(obj)
        parsed = extract_json_object(text)
        if parsed is not None:
            return parsed
        if isinstance(obj, dict):
            parsed = extract_json_object(json.dumps(obj, ensure_ascii=False))
            if parsed is not None:
                return parsed
        raise ValueError("LLM response did not contain a JSON object")

    parts: List[str] = []
    for raw in response.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line == "[DONE]":
            break
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunk = _extract_chunk(obj)
        if chunk:
            parts.append(chunk)
        if obj.get("done") is True:
            break

    parsed = extract_json_object("".join(parts))
    if parsed is None:
        raise ValueError("streamed LLM response did not contain a JSON object")
    return parsed
