from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, gold: int, pred: int) -> None:
        if gold == 1 and pred == 1:
            self.tp += 1
        elif gold == 0 and pred == 1:
            self.fp += 1
        elif gold == 0 and pred == 0:
            self.tn += 1
        elif gold == 1 and pred == 0:
            self.fn += 1


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_binary_metrics(records: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    counts = BinaryCounts()
    total = 0
    skipped = 0
    for record in records:
        try:
            gold = int(record.get("gold_label"))
            pred = int(record.get("pred_label"))
        except (TypeError, ValueError):
            skipped += 1
            continue
        if gold not in (0, 1) or pred not in (0, 1):
            skipped += 1
            continue
        counts.update(gold, pred)
        total += 1

    precision = safe_div(counts.tp, counts.tp + counts.fp)
    recall = safe_div(counts.tp, counts.tp + counts.fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(counts.tp + counts.tn, total)
    return {
        **asdict(counts),
        "total": total,
        "skipped": skipped,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def group_records(
    records: Iterable[Mapping[str, object]], group_fields: Sequence[str]
) -> Dict[Tuple[str, ...], List[Mapping[str, object]]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, object]]] = {}
    for record in records:
        key = tuple(str(record.get(field, "")) for field in group_fields)
        groups.setdefault(key, []).append(record)
    return groups
