from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _escape_latex(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _format_cell(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    text = str(value)
    try:
        num = float(text)
    except ValueError:
        return _escape_latex(text)
    if text.strip() == "":
        return ""
    if abs(num) <= 1.0 and "." in text:
        return f"{num:.{digits}f}"
    if num.is_integer():
        return str(int(num))
    return f"{num:.{digits}f}"


def rows_to_latex(
    rows: Sequence[dict],
    columns: Sequence[str] | None = None,
    caption: str = "",
    label: str = "",
    digits: int = 4,
) -> str:
    if not rows:
        columns = list(columns or [])
    elif columns is None:
        columns = list(rows[0].keys())
    else:
        columns = list(columns)

    alignment = "l" * max(len(columns), 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(_escape_latex(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_format_cell(row.get(c), digits) for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if caption:
        lines.append(rf"\caption{{{_escape_latex(caption)}}}")
    if label:
        lines.append(rf"\label{{{_escape_latex(label)}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def csv_to_latex(
    input_path: Path,
    output_path: Path,
    columns: Sequence[str] | None = None,
    caption: str = "",
    label: str = "",
    digits: int = 4,
) -> None:
    rows = read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        rows_to_latex(rows, columns=columns, caption=caption, label=label, digits=digits),
        encoding="utf-8",
    )
