import glob
import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import spacy
import tyro
from constant import SPATIAL_PHRASES, SPATIAL_SINGLE, STYLISTIC_PHRASES, STYLISTIC_SINGLE
from datasets import load_dataset
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from spacy.matcher import PhraseMatcher

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science"])
    plt.rcParams["font.family"] = "Times New Roman"
except Exception:
    pass

nlp = spacy.load("en_core_web_sm", disable=["ner"])
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
phrase_matcher.add("SPATIAL", [nlp.make_doc(p) for p in SPATIAL_PHRASES])
phrase_matcher.add("STYLISTIC", [nlp.make_doc(p) for p in STYLISTIC_PHRASES])
SPATIAL_SET = set(SPATIAL_SINGLE)
STYLISTIC_SET = set(STYLISTIC_SINGLE)


def tag_prompt(prompt: str):
    """
    Returns per-category counts and token-level assignments using precedence:
    SPATIAL > STYLISTIC. Denominator = tokens (not stopwords/punct).
    """
    doc = nlp(prompt)
    N = len(doc)

    # mark multi-word phrase spans first
    spans = phrase_matcher(doc)
    covered = [None] * N

    # apply phrase tags with precedence
    for match_id, start, end in spans:
        label = nlp.vocab.strings[match_id]  # "SPATIAL" or "STYLISTIC"
        for i in range(start, end):
            covered[i] = label

    # single-token tagging
    for i, tok in enumerate(doc):
        if tok.is_punct or tok.is_space:
            continue
        if covered[i] is not None:
            continue
        lemma = tok.lemma_.lower()
        if lemma in SPATIAL_SET:
            covered[i] = "SPATIAL"
        elif lemma in STYLISTIC_SET:
            covered[i] = "STYLISTIC"

    # build counts
    cat_counts = Counter([c for i, c in enumerate(covered) if c])
    denom = sum(c is not None for c in covered)
    densities = {c: (cat_counts[c] / denom if denom else 0.0) for c in ["SPATIAL", "STYLISTIC"]}

    return {
        "counts": dict(cat_counts),
        "denom_tokens": denom,
        "densities": densities,
        "balance_index": entropy_norm([densities.get("SPATIAL", 0.0), densities.get("STYLISTIC", 0.0)]),
    }


def entropy_norm(p: list[float]) -> float:
    """Normalized entropy (0..1) for a probability-like list p (sum<=1 ok)."""
    s = sum(p)
    if s <= 0:
        return 0.0
    q = [x / s for x in p if x > 0]
    H = -sum(x * math.log(x + 1e-12) for x in q)
    return float(H)


def prep_stats(text: str) -> tuple[int, int, float]:
    doc = nlp(text)
    relations = sum(1 for tok in doc if tok.dep_ == "prep")
    return relations


def coverage(
    prompts: list[str],
    tau_spa: float = 0.05,
    tau_sty: float = 0.05,
    min_spa: int = 5,
    min_sty: int = 5,
):
    """
    A prompt 'covers' a category if it passes EITHER a density threshold OR a raw count threshold.
    Tune thresholds to your corpus length.
    """
    stats = [tag_prompt(p) for p in prompts]

    def pass_cat(s, cat, tau, kmin):
        dens = s["densities"].get(cat, 0.0)
        cnt = s["counts"].get(cat, 0)
        return (dens >= tau) or (cnt >= kmin)

    spa_cov = sum(pass_cat(s, "SPATIAL", tau_spa, min_spa) for s in stats) / len(stats)
    sty_cov = sum(pass_cat(s, "STYLISTIC", tau_sty, min_sty) for s in stats) / len(stats)
    return {
        "spatial_coverage": max(spa_cov, 0.0),
        "stylistic_coverage": max(sty_cov, 0.0),
        "avg_balance_index": max(sum(s["balance_index"] for s in stats) / len(stats), 0.0),
    }


def count_average_length(prompts: list[str]):
    total_length = 0
    for prompt in prompts:
        total_length += len(prompt)
    return total_length / len(prompts)


def count_relations(prompts: list[str]):
    total_relations = 0
    for prompt in prompts:
        total_relations += prep_stats(prompt)
    return total_relations


def get_statistics_from_prompts(prompts: list[str]):
    avg_length = count_average_length(prompts)
    avg_relations = count_relations(prompts)
    coverage_stats = coverage(prompts)
    return {
        "avg_length": avg_length,
        "avg_relations": avg_relations,
        "spatial_coverage": coverage_stats["spatial_coverage"],
        "stylistic_coverage": coverage_stats["stylistic_coverage"],
        "avg_balance_index": coverage_stats["avg_balance_index"],
    }


def count_t2i_compbench(data_root_dir: str) -> list[str]:
    data_files = glob.glob(os.path.join(data_root_dir, "*.txt"))
    prompts = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            data = f.readlines()
        for line in data:
            if line.strip() == "":
                continue
            prompts.append(line.strip())
    return prompts


def count_geneval(data_root_dir: str) -> list[str]:
    prompts = []
    with open(os.path.join(data_root_dir, "prompts.txt"), "r") as f:
        data = f.readlines()
    for line in data:
        if line.strip() == "":
            continue
        prompts.append(line.strip())
    return prompts


def count_dpg_bench(data_root_dir: str) -> list[str]:
    data_files = glob.glob(os.path.join(data_root_dir, "*.txt"))
    prompts = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            data = f.readlines()
        for line in data:
            if line.strip() == "":
                continue
        prompts.append(line.strip())
    return prompts


def count_rarebench(data_root_dir: str) -> list[str]:
    data_files = glob.glob(os.path.join(data_root_dir, "*.txt"))  # although .txt, it is a .json file
    prompts = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            data = json.load(f)
        for prompt in data.keys():
            prompts.append(prompt.strip())
    return prompts


def count_genai_bench(*_, **__) -> list[str]:
    dataset = load_dataset("BaiqiL/GenAI-Bench")
    prompts = [example["Prompt"] for example in dataset["train"]]
    del dataset
    return prompts


def count_lpd_bench(data_root_dir: str) -> list[str]:
    data_files = glob.glob(os.path.join(data_root_dir, "*.json"))
    prompts = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            data = json.load(f)
        prompts.append(data["prompt"].strip())
    return prompts


def verbose_print(stats: dict, dataset_name: str):
    print("================================================")
    print(f"{dataset_name} statistics:")
    print("================================================")
    print(f"Average length: {stats['avg_length']}")
    print(f"Average relations: {stats['avg_relations']}")
    print(f"Spatial coverage: {stats['spatial_coverage']}")
    print(f"Stylistic coverage: {stats['stylistic_coverage']}")
    print(f"Average balance index: {stats['avg_balance_index']}")


target_dataset = {
    "t2i_compbench": (count_t2i_compbench, "T2I-CompBench"),
    "geneval": (count_geneval, "GenEval"),
    "dpg_bench": (count_dpg_bench, "DPG-Bench"),
    "genai_bench": (count_genai_bench, "GenAI-Bench"),
    "lpd_bench": (count_lpd_bench, r"\textbf{LPD-Bench}"),
}


def main(
    data_root_dir: str = "data",
    verbose: bool = False,
    plot: bool = False,
    plot_save_path: str = "assets/dataset_statistics.pdf",
):
    metrics = {}
    for dataset_name, (count_func, dataset_display_name) in target_dataset.items():
        prompts = count_func(os.path.join(data_root_dir, dataset_name))
        stats = get_statistics_from_prompts(prompts)
        if verbose:
            verbose_print(stats, dataset_display_name)
        metrics[dataset_display_name] = stats

    if plot:
        dataset_names = list(metrics.keys())
        sort_index = sorted(range(len(dataset_names)), key=lambda i: metrics[dataset_names[i]]["avg_length"])
        dataset_names = [dataset_names[i] for i in sort_index]
        avg_lengths = [metrics[dataset_name]["avg_length"] for dataset_name in dataset_names]
        spatial_coverages = [metrics[dataset_name]["spatial_coverage"] for dataset_name in dataset_names]
        stylistic_coverages = [metrics[dataset_name]["stylistic_coverage"] for dataset_name in dataset_names]
        balance_indices = [metrics[dataset_name]["avg_balance_index"] for dataset_name in dataset_names]

        dataset_colors = dict(
            zip(dataset_names, plt.cm.Blues(np.linspace(0.4, 0.8, len(dataset_names))), strict=True)
        )

        # Create figure with GridSpec
        # Adjusted width ratios to 2:1 for coverage:balance
        fig = plt.figure(figsize=(7, 3.2), constrained_layout=False)
        gs = GridSpec(
            nrows=3,
            ncols=2,
            height_ratios=[1, 0.05, 2.5],
            width_ratios=[2, 1],
            hspace=-0.1,
            wspace=0.05,
            figure=fig,
        )

        # Top subplot - Average Length (spans both columns)
        ax_top = fig.add_subplot(gs[0, :])

        avg_lengths_int = [int(round(v)) for v in avg_lengths]
        colors_sorted = [dataset_colors[dataset_name] for dataset_name in dataset_names]

        original_left = left = 850
        ax_top.text(
            0.01,
            0.5,
            "(a) Avg. Length",
            transform=ax_top.transAxes,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
            fontfamily="times new roman",
        )
        bars_top = []
        for target_id, (value, dataset, color) in enumerate(
            zip(avg_lengths_int, dataset_names, colors_sorted, strict=True)
        ):
            b = ax_top.barh(0, value, left=left, color=color, height=0.5, label=dataset)
            bars_top.append(b[0])

            if target_id in [0, 1, 2]:
                left += value
                continue
            plot_value = value
            ax_top.text(
                left + value / 2,
                0,
                f"{plot_value}",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
            )
            left += value

        bar_height = 0.5
        ax_top.add_patch(
            Rectangle(
                (original_left, -bar_height / 2),
                left - original_left,
                bar_height * 1.01,
                linewidth=0.8,
                edgecolor="black",
                facecolor="none",
                zorder=10,
                clip_on=False,
            )
        )

        ax_top.axis("off")
        ax_top.set_xlim(0, left + 2)
        ax_top.set_ylim(-0.85, 0.85)

        # LEFT SUBPLOT - Coverage Metrics
        ax_left = fig.add_subplot(gs[2, 0])

        coverage_metrics = ["(b.1) Spatial Coverage", "(b.2) Stylistic Coverage"]
        coverage_data = [spatial_coverages, stylistic_coverages]

        x_left = np.arange(len(coverage_metrics))
        bar_width = 0.14

        bars_for_legend = []
        labels_for_legend = []

        for i, dataset in enumerate(dataset_names):
            vals = [coverage_data[m][i] for m in range(len(coverage_metrics))]
            color = dataset_colors[dataset]
            offsets = x_left + (i - len(dataset_names) / 2) * bar_width + bar_width / 2
            bars = ax_left.bar(offsets, vals, width=bar_width, color=color, label=dataset)
            for bar, val in zip(bars, vals, strict=True):
                ax_left.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            bars_for_legend.append(bars[0])
            labels_for_legend.append(dataset)

        ax_left.set_xticks(x_left)
        ax_left.set_xticklabels(coverage_metrics, fontsize=11, fontfamily="times new roman")
        ax_left.set_yticks([])
        ax_left.set_ylim(0, 1.1)
        ax_left.set_xlim(-0.4, len(coverage_metrics) - 0.6)
        ax_left.grid(axis="y", linestyle="--", alpha=0.5)
        ax_left.set_axisbelow(True)

        # RIGHT SUBPLOT - Balance Score
        ax_right = fig.add_subplot(gs[2, 1])

        balance_metrics = ["(c) Balance Score"]
        balance_data = [balance_indices]

        x_right = np.arange(len(balance_metrics))

        for i, dataset in enumerate(dataset_names):
            vals = [balance_data[m][i] for m in range(len(balance_metrics))]
            color = dataset_colors[dataset]
            offsets = x_right + (i - len(dataset_names) / 2) * bar_width + bar_width / 2
            bars = ax_right.bar(offsets, vals, width=bar_width, color=color, label=dataset)
            for bar, val in zip(bars, vals, strict=True):
                ax_right.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontfamily="times new roman",
                )

        ax_right.set_xticks(x_right)
        ax_right.set_xticklabels(balance_metrics, fontsize=11)
        ax_right.set_yticks([])
        ax_right.set_ylim(0, 1.1)
        ax_right.set_xlim(-0.4, len(balance_metrics) - 0.6)
        ax_right.grid(axis="y", linestyle="--", alpha=0.5)
        ax_right.set_axisbelow(True)

        fig.legend(
            handles=bars_for_legend,
            labels=labels_for_legend,
            loc="lower center",
            bbox_to_anchor=(0.515, 0.82),
            ncol=len(dataset_names),
            frameon=False,
            fontsize=9,
            handlelength=0.8,
            handleheight=0.8,
            columnspacing=1.5,
        )

        fig.subplots_adjust(bottom=0.11)
        fig.savefig(plot_save_path, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    tyro.cli(main)
