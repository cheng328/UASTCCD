from __future__ import annotations

from typing import Iterable, Tuple


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def compute_length_stats(texts: Iterable[str]) -> Tuple[int, float, int]:
    values = [count_tokens(t) for t in texts]
    if not values:
        return 0, 0.0, 0
    return max(values), sum(values) / len(values), len(values)

