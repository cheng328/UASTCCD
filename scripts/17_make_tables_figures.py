from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.figure_writer import plot_k_sensitivity
from src.evaluation.table_writer import csv_to_latex


DEFAULT_TABLES = {
    "table4_clcd_cross_language.csv": ("table4_clcd_cross_language.tex", "CLCD cross-language results.", "tab:clcd"),
    "table5_single_language.csv": ("table5_single_language.tex", "Single-language clone detection results.", "tab:single"),
    "table6_evidence_llm_robustness.csv": (
        "table6_evidence_llm_robustness.tex",
        "Evidence LLM robustness results.",
        "tab:evidence-llm",
    ),
    "table7_mapper_ablation.csv": ("table7_mapper_ablation.tex", "Learnable mapper ablation results.", "tab:mapper-ablation"),
    "table8_workflow_stability.csv": ("table8_workflow_stability.tex", "Workflow stability results.", "tab:workflow"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables and figures from result CSVs.")
    parser.add_argument("--input", default="", help="Single CSV input")
    parser.add_argument("--output", default="", help="Single output path")
    parser.add_argument("--kind", default="table", choices=["table", "figure2", "all"])
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--caption", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--columns", default="", help="Comma-separated columns for table output")
    return parser.parse_args()


def make_table(input_path: Path, output_path: Path, caption: str = "", label: str = "", columns: str = "") -> None:
    cols = [c.strip() for c in columns.split(",") if c.strip()] or None
    csv_to_latex(input_path, output_path, columns=cols, caption=caption, label=label)
    print(f"Wrote: {output_path}")


def make_all(results_dir: Path) -> None:
    for csv_name, (tex_name, caption, label) in DEFAULT_TABLES.items():
        csv_path = results_dir / csv_name
        if csv_path.exists():
            make_table(csv_path, results_dir / tex_name, caption, label)

    fig_csv = results_dir / "figure2_k_sensitivity.csv"
    if fig_csv.exists():
        try:
            plot_k_sensitivity(fig_csv, results_dir / "figure2_k_sensitivity.pdf")
            print(f"Wrote: {results_dir / 'figure2_k_sensitivity.pdf'}")
        except RuntimeError as exc:
            print(str(exc))


def main() -> int:
    args = parse_args()
    if args.kind == "all":
        make_all(Path(args.results_dir))
        return 0

    if not args.input or not args.output:
        raise SystemExit("--input and --output are required unless --kind all")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.kind == "table":
        make_table(input_path, output_path, args.caption, args.label, args.columns)
        return 0
    if args.kind == "figure2":
        try:
            plot_k_sensitivity(input_path, output_path)
            print(f"Wrote: {output_path}")
        except RuntimeError as exc:
            print(str(exc))
        return 0
    raise SystemExit(f"Unsupported kind: {args.kind}")


if __name__ == "__main__":
    raise SystemExit(main())
