from __future__ import annotations

import csv
from pathlib import Path
from typing import List


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_k_sensitivity(input_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.with_suffix(".txt").write_text(
            "matplotlib is not installed; cannot render Figure 2 locally.\n"
            f"Input CSV: {input_path}\n",
            encoding="utf-8",
        )
        raise RuntimeError("matplotlib is not installed") from exc

    rows = read_csv(input_path)
    k_values = [int(float(row["K"])) for row in rows]
    f1_values = [float(row["f1"] if "f1" in row else row["F1"]) for row in rows]
    active_values = [
        float(row.get("active_categories") or row.get("Active Categories") or 0) for row in rows
    ]

    fig, ax1 = plt.subplots(figsize=(5.2, 3.2))
    ax1.plot(k_values, f1_values, marker="o", label="F1", color="#1f77b4")
    ax1.set_xlabel("K")
    ax1.set_ylabel("F1")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(k_values, active_values, marker="s", label="Active categories", color="#2ca02c")
    ax2.set_ylabel("Active categories")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
