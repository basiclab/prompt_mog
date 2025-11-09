import glob
import json
import os
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science"])
    plt.rcParams["font.family"] = "Times New Roman"
except Exception:
    pass

from gen_utils.common import create_pipeline, setup_logging

model_list = [
    ("sd3", "stabilityai/stable-diffusion-3.5-large"),
    ("flux", "black-forest-labs/FLUX.1-Krea-dev"),
    ("cogview4", "THUDM/CogView4-6B"),
    ("qwen", "Qwen/Qwen-Image"),
]


def main(
    data_root: str = "data/lpbench/filtered",
    config_root: str = "configs",
    num_runs: int = 5,
    model_list: list[tuple[str, str]] = model_list,
    plot: bool = False,
    plot_save_path: str = "assets/model_runtime.pdf",
):
    selected_prompt = None  # the longest one

    for file in glob.glob(os.path.join(data_root, "*.json")):
        with open(file, "r") as f:
            prompt_data = json.load(f)
        if selected_prompt is None or len(prompt_data["prompt"]) > len(selected_prompt):
            selected_prompt = prompt_data["prompt"]

    print(f"Length of selected prompt: {len(selected_prompt)}")

    generator = torch.Generator(device="cpu").manual_seed(42)

    runtime_results = defaultdict(dict)

    for model_name, pretrained_name in model_list:
        print(f"\n{'=' * 60}")
        print(f"Testing {model_name}")
        print(f"{'=' * 60}")

        config_path = os.path.join(
            config_root, "gen", f"{os.path.basename(pretrained_name).replace('-', '_').lower()}.json"
        )
        with open(config_path, "r") as f:
            gen_params = json.load(f)

        for model_type in ["short", "pmog"]:
            print(f"\n--- Mode: {model_type} ---")
            pipe = create_pipeline(pretrained_name, torch.bfloat16, False, "cuda", model_type)
            pipe.set_progress_bar_config(disable=True)

            if model_type == "pmog":
                # Gamma and sigma are dummy variable that does not affect the runtime
                # num_mode is the default setup for our experiments
                pipe.encode_prompt = partial(
                    pipe.encode_prompt,
                    gamma=1,
                    num_mode=50,
                    sigma=0.5,
                    generator=generator,  # batch size is 1 for now
                )

            # Warmup run
            print("Warmup run...")
            pipe(
                prompt=selected_prompt,
                **gen_params,
            )
            torch.cuda.synchronize()

            runtimes = []
            for run_idx in range(num_runs):
                print(f"Run {run_idx + 1}/{num_runs}...", end=" ")

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                pipe(
                    prompt=selected_prompt,
                    **gen_params,
                )

                end_event.record()

                torch.cuda.synchronize()

                runtime = start_event.elapsed_time(end_event) / 1000.0
                runtimes.append(runtime)
                print(f"{runtime:.2f}s")

            avg_runtime = sum(runtimes) / len(runtimes)
            min_runtime = min(runtimes)
            max_runtime = max(runtimes)

            runtime_results[model_name][model_type] = {
                "avg": avg_runtime,
                "min": min_runtime,
                "max": max_runtime,
                "all": runtimes,
            }

            print(f"Average: {avg_runtime:.2f}s, Min: {min_runtime:.2f}s, Max: {max_runtime:.2f}s")

            del pipe
            torch.cuda.empty_cache()

    # Print summary
    print(f"\n\n{'=' * 60}")
    print("RUNTIME COMPARISON SUMMARY")
    print(f"{'=' * 60}\n")

    for model_name, modes in runtime_results.items():
        print(f"{model_name.upper()}:")
        if "short" in modes and "pmog" in modes:
            short_avg = modes["short"]["avg"]
            pmog_avg = modes["pmog"]["avg"]
            speedup = short_avg / pmog_avg if pmog_avg > 0 else 0
            diff_percent = ((pmog_avg - short_avg) / short_avg * 100) if short_avg > 0 else 0

            print(f"  Short mode: {short_avg:.2f}s (±{modes['short']['max'] - modes['short']['min']:.2f}s)")
            print(f"  PMOG mode:  {pmog_avg:.2f}s (±{modes['pmog']['max'] - modes['pmog']['min']:.2f}s)")
            print(f"  Difference: {diff_percent:+.1f}% ({pmog_avg - short_avg:+.2f}s)")
            print(f"  Speedup:    {speedup:.2f}x {'(Short faster)' if speedup < 1 else '(PMOG faster)'}")
        print()

    models = list(runtime_results.keys())
    x = np.arange(len(models))
    width = 0.35

    short_times = []
    pmog_times = []

    for model_name in models:
        short_times.append(runtime_results[model_name]["short"]["avg"])
        pmog_times.append(runtime_results[model_name]["pmog"]["avg"])

    if plot:
        fig, ax = plt.subplots(figsize=(8, 3))

        bars1 = ax.bar(x - width / 2, short_times, width, label="Original", color="royalblue", alpha=0.8)
        bars2 = ax.bar(x + width / 2, pmog_times, width, label="+ PMoG", color="lightcoral", alpha=0.8)

        for i, (bar1, bar2) in enumerate(zip(bars1, bars2, strict=True)):
            short_time = short_times[i]
            pmog_time = pmog_times[i]

            ax.text(
                bar1.get_x() + bar1.get_width() / 2,
                bar1.get_height() + max(short_times) * 0.02,
                f"{short_time:.2f}s",
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
            )

            diff_percent = ((pmog_time - short_time) / short_time * 100) if short_time > 0 else 0
            label_text = f"{pmog_time:.2f}s\n(${{\\scriptsize {diff_percent:+.1f}\\%}}$)"
            ax.text(
                bar2.get_x() + bar2.get_width() / 2,
                bar2.get_height() + max(pmog_times) * 0.02,
                label_text,
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="bold",
            )

        ax.set_yticks([])
        ax.set_xticks(x)
        ax.set_xticklabels(["SD3.5-Large", "Flux.1-Krea.Dev", "CogView4-6B", "Qwen-Image"], fontsize=14)
        ax.legend(fontsize=12, ncols=2, loc="upper center")

        y_max = max(max(short_times), max(pmog_times))
        ax.set_ylim(0, y_max * 1.25)

        plt.tight_layout()
        plt.savefig(plot_save_path, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    setup_logging()
    tyro.cli(main)
