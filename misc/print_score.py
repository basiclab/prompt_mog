import json
import sys
from pathlib import Path
from statistics import mean, stdev

import tyro

COLS = [
    ("semantic", "clip_score"),
    ("semantic", "vqa_score"),
    ("spatial", "clip_score"),
    ("spatial", "vqa_score"),
    ("stylistic", "clip_score"),
    ("stylistic", "vqa_score"),
    ("average", "clip_score"),
    ("average", "vqa_score"),
]
HEADER_LABELS = [
    "semantic-clip",
    "semantic-vqa",
    "spatial-clip",
    "spatial-vqa",
    "stylistic-clip",
    "stylistic-vqa",
    "average-clip",
    "average-vqa",
]


def load_seed_scores(seed_dir: Path):
    fp = seed_dir / "average_score.json"
    if not fp.is_file():
        return None
    try:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        s_clip = float(data["semantic"]["clip_score"]) * 100.0
        s_vqa = float(data["semantic"]["vqa_score"]) * 100.0
        sp_clip = float(data["spatial"]["clip_score"]) * 100.0
        sp_vqa = float(data["spatial"]["vqa_score"]) * 100.0
        st_clip = float(data["stylistic"]["clip_score"]) * 100.0
        st_vqa = float(data["stylistic"]["vqa_score"]) * 100.0

        avg_clip = (s_clip + sp_clip + st_clip) / 3.0
        avg_vqa = (s_vqa + sp_vqa + st_vqa) / 3.0

        return {
            ("semantic", "clip_score"): s_clip,
            ("semantic", "vqa_score"): s_vqa,
            ("spatial", "clip_score"): sp_clip,
            ("spatial", "vqa_score"): sp_vqa,
            ("stylistic", "clip_score"): st_clip,
            ("stylistic", "vqa_score"): st_vqa,
            ("average", "clip_score"): avg_clip,
            ("average", "vqa_score"): avg_vqa,
        }
    except Exception as e:
        print(f"[warn] Skipping {fp}: {e}", file=sys.stderr)
        return None


def find_seed_dirs(root: Path):
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "average_score.json").exists():
            yield p


def label_for_seed(dir_name: str):
    name = dir_name.strip().lower()
    num = None
    for part in name.replace("-", "_").split("_"):
        if part.isdigit():
            num = int(part)
    if name.startswith("seed") and num is not None:
        return f"seed{num}"
    if num is not None:
        return f"seed{num}"
    return dir_name


def seed_sort_key(name):
    low = name.lower()
    if low.startswith("seed"):
        num = "".join(ch for ch in low[4:] if ch.isdigit())
        if num.isdigit():
            return (0, int(num))
    return (1, low)


def format_plain(rows, col_means, col_stds, show_labels=True):
    w0 = 8
    w = 15

    lines = []
    if show_labels:
        header = f"{'':<{w0}}" + "".join(f"{h:>{w}}" for h in HEADER_LABELS)
        lines.append(header)
        lines.append("")  # blank line

    def fmt_vals(values):
        return "".join(f"{v:>{w}.2f}" for v in values)

    for label, vals in sorted(rows, key=lambda x: seed_sort_key(x[0])):
        if show_labels:
            lines.append(f"{label:<{w0}}{fmt_vals(vals)}")
        else:
            lines.append(fmt_vals(vals).lstrip())  # no leading label column

    # summary
    lines.append("")
    if show_labels:
        lines.append(f"{'mean':<{w0}}{fmt_vals(col_means)}")
        lines.append(f"{'std':<{w0}}{fmt_vals(col_stds)}")
    else:
        lines.append(fmt_vals(col_means).lstrip())
        lines.append(fmt_vals(col_stds).lstrip())

    return "\n".join(lines)


def format_latex(rows, col_means, col_stds, show_labels=True):
    def join_cells(cells):
        return " & ".join(cells) + r" \\"

    lines = []
    # Header
    if show_labels:
        lines.append(join_cells([""] + HEADER_LABELS))
    else:
        lines.append(join_cells(HEADER_LABELS))

    # Seed rows
    def fmt_num(x):  # 6 decimals
        return f"{x:.2f}"

    for label, vals in sorted(rows, key=lambda x: seed_sort_key(x[0])):
        cells = [fmt_num(v) for v in vals]
        if show_labels:
            cells = [label] + cells
        lines.append(join_cells(cells))

    # Summary row (mean ± std)
    def fmt_mean_std(m, s):
        return f"{m:.2f} {{\\scriptsize $\\pm$ {s:.2f}}}"

    summary_cells = [fmt_mean_std(m, s) for m, s in zip(col_means, col_stds, strict=True)]
    if show_labels:
        summary_cells = ["mean"] + summary_cells
    lines.append(join_cells(summary_cells))

    return "\n".join(lines)


def main(
    root: Path,
    label: bool = True,
    latex: bool = False,
):
    root = root.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for seed_dir in find_seed_dirs(root):
        scores = load_seed_scores(seed_dir)
        if scores is None:
            continue
        seed_label = label_for_seed(seed_dir.name)
        row_vals = [scores[key] for key in COLS]
        rows.append((seed_label, row_vals))

    if not rows:
        print("No seeds found with average_score.json files.", file=sys.stderr)
        sys.exit(2)

    # Column-wise mean/std
    cols_transposed = list(zip(*[vals for _, vals in rows], strict=True))
    col_means = [mean(col) for col in cols_transposed]
    if len(rows) >= 2:
        col_stds = [stdev(col) for col in cols_transposed]
    else:
        col_stds = [0.0 for _ in cols_transposed]

    # Output
    if latex:
        out = format_latex(rows, col_means, col_stds, show_labels=label)
    else:
        out = format_plain(rows, col_means, col_stds, show_labels=label)
    print(out)


if __name__ == "__main__":
    tyro.cli(main)
