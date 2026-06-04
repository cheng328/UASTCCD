from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


CLCD_LANGUAGE_PAIRS = [
    "c++-c#",
    "java-c#",
    "java-c++",
    "java-python",
    "python-c#",
    "python-c++",
]

SINGLE_LANGUAGE_DATASETS = ["gcj", "ojclone", "bigclonebench"]

DATASET_ALIASES = {
    "googlejam4": "gcj",
    "googlejam": "gcj",
    "bcb": "bigclonebench",
    "bigclonebench": "bigclonebench",
    "ojclone": "ojclone",
    "gcj": "gcj",
}


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({str(k).lstrip("\ufeff").strip(): v for k, v in row.items()})
        return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_triplet(row: dict) -> str:
    if not row:
        return ""
    return f"{float(row.get('precision', 0.0)):.4f}/{float(row.get('recall', 0.0)):.4f}/{float(row.get('f1', 0.0)):.4f}"


def aggregate_clcd_table(metrics_rows: Iterable[dict], language_pairs: Sequence[str] = CLCD_LANGUAGE_PAIRS) -> List[Dict[str, object]]:
    by_method_pair: Dict[tuple, dict] = {}
    methods = []
    for row in metrics_rows:
        method = str(row.get("method", ""))
        pair = str(row.get("language_pair", "")).lower()
        if method and method not in methods:
            methods.append(method)
        by_method_pair[(method, pair)] = row

    rows: List[Dict[str, object]] = []
    for method in methods:
        out: Dict[str, object] = {"method": method}
        f1_values = []
        for pair in language_pairs:
            row = by_method_pair.get((method, pair), {})
            out[pair] = _metric_triplet(row)
            if row:
                try:
                    f1_values.append(float(row.get("f1", 0.0)))
                except ValueError:
                    pass
        out["macro_f1"] = sum(f1_values) / len(f1_values) if f1_values else 0.0
        rows.append(out)
    return rows


def aggregate_single_language_table(
    metrics_rows: Iterable[dict], datasets: Sequence[str] = SINGLE_LANGUAGE_DATASETS
) -> List[Dict[str, object]]:
    by_method_dataset: Dict[tuple, dict] = {}
    methods = []
    for row in metrics_rows:
        method = str(row.get("method", ""))
        dataset_raw = str(row.get("dataset", "")).lower()
        dataset = DATASET_ALIASES.get(dataset_raw, dataset_raw)
        if method and method not in methods:
            methods.append(method)
        by_method_dataset[(method, dataset)] = row

    rows: List[Dict[str, object]] = []
    for method in methods:
        out: Dict[str, object] = {"method": method}
        f1_values = []
        for dataset in datasets:
            row = by_method_dataset.get((method, dataset), {})
            out[dataset] = _metric_triplet(row)
            if row:
                try:
                    f1_values.append(float(row.get("f1", 0.0)))
                except ValueError:
                    pass
        out["macro_f1"] = sum(f1_values) / len(f1_values) if f1_values else 0.0
        rows.append(out)
    return rows
