import gc
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
    num_images_per_prompt: int = 6,
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

        for model_type in ["short", "pmog", "cads", "df"]:
            print(f"\n--- Pipeline Type: {model_type} ---")
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
                num_images_per_prompt=num_images_per_prompt,
            )
            torch.cuda.synchronize()

            # Timed runs using CUDA events
            runtimes = []
            for run_idx in range(num_runs):
                print(f"Run {run_idx + 1}/{num_runs}...", end=" ")

                # Create CUDA events for timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Record start time
                start_event.record()

                pipe(
                    prompt=selected_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    **gen_params,
                )

                # Record end time
                end_event.record()

                # Wait for completion
                torch.cuda.synchronize()

                # Get elapsed time in milliseconds and convert to seconds
                runtime = start_event.elapsed_time(end_event) / 1000.0
                runtime = runtime / num_images_per_prompt
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
            gc.collect()

    # Print summary
    print(f"\n\n{'=' * 60}")
    print("RUNTIME COMPARISON SUMMARY")
    print(f"{'=' * 60}\n")

    for model_name, modes in runtime_results.items():
        print(f"{model_name.upper()}:")
        if "short" in modes and "pmog" in modes:
            short_avg = modes["short"]["avg"]
            pmog_avg = modes["pmog"]["avg"]
            cads_avg = modes["cads"]["avg"]
            df_avg = modes["df"]["avg"]
            pmog_speedup = short_avg / pmog_avg if pmog_avg > 0 else 0
            cads_speedup = short_avg / cads_avg if cads_avg > 0 else 0
            df_speedup = short_avg / df_avg if df_avg > 0 else 0

            print(f"  Short mode: {short_avg:.2f}s (±{modes['short']['max'] - modes['short']['min']:.2f}s)")
            print(f"  PMOG mode:  {pmog_avg:.2f}s (±{modes['pmog']['max'] - modes['pmog']['min']:.2f}s)")
            print(f"  CADS mode:  {cads_avg:.2f}s (±{modes['cads']['max'] - modes['cads']['min']:.2f}s)")
            print(f"  Diverse Flow mode: {df_avg:.2f}s (±{modes['df']['max'] - modes['df']['min']:.2f}s)")
            print(
                f"  PMOG speed: {pmog_speedup:.2f}x {'(Short faster)' if pmog_speedup < 1 else '(PMOG faster)'}"
            )
            print(
                f"  CADS speed: {cads_speedup:.2f}x {'(Short faster)' if cads_speedup < 1 else '(CADS faster)'}"
            )
            print(
                f"  Diverse Flow speed: {df_speedup:.2f}x {'(Short faster)' if df_speedup < 1 else '(Diverse Flow faster)'}"
            )
        print()

    if plot:
        models = list(runtime_results.keys())
        x = np.arange(len(models))

        short_times = []
        pmog_times = []
        cads_times = []
        df_times = []

        for model_name in models:
            short_times.append(runtime_results[model_name]["short"]["avg"])
            pmog_times.append(runtime_results[model_name]["pmog"]["avg"])
            cads_times.append(runtime_results[model_name]["cads"]["avg"])
            df_times.append(runtime_results[model_name]["df"]["avg"])

        fig, ax = plt.subplots(figsize=(8, 3))

        width = 0.18

        bars1 = ax.bar(x - 1.5 * width, short_times, width, label="Short", color="royalblue", alpha=0.8)
        bars2 = ax.bar(x - 0.5 * width, pmog_times, width, label="+ PMoG", color="lightcoral", alpha=0.8)
        bars3 = ax.bar(x + 0.5 * width, cads_times, width, label="+ CADS", color="lightgreen", alpha=0.8)
        bars4 = ax.bar(
            x + 1.5 * width, df_times, width, label="+ Diverse Flow", color="lightyellow", alpha=0.8
        )

        y_max = max(max(short_times), max(pmog_times), max(cads_times), max(df_times))

        for i, (bar1, bar2, bar3, bar4) in enumerate(zip(bars1, bars2, bars3, bars4, strict=True)):
            s = short_times[i]
            p = pmog_times[i]
            c = cads_times[i]
            d = df_times[i]

            ax.text(
                bar1.get_x() + bar1.get_width() / 2,
                bar1.get_height() + y_max * 0.02,
                f"{s:.2f}s",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

            pmog_diff = ((p - s) / s * 100) if s > 0 else 0.0
            cads_diff = ((c - s) / s * 100) if s > 0 else 0.0
            df_diff = ((d - s) / s * 100) if s > 0 else 0.0

            ax.text(
                bar2.get_x() + bar2.get_width() / 2,
                bar2.get_height() + y_max * 0.02,
                f"{p:.2f}s\n({pmog_diff:+.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
            ax.text(
                bar3.get_x() + bar3.get_width() / 2,
                bar3.get_height() + y_max * 0.02,
                f"{c:.2f}s\n({cads_diff:+.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
            ax.text(
                bar4.get_x() + bar4.get_width() / 2,
                bar4.get_height() + y_max * 0.02,
                f"{d:.2f}s\n({df_diff:+.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        display_names = {
            "sd3.5-large": "SD3.5-Large",
            "flux.1-krea-dev": "Flux.1-Krea.Dev",
            "cogview4-6b": "CogView4-6B",
            "qwen-image": "Qwen-Image",
        }

        ax.set_yticks([])
        ax.set_xticks(x)
        ax.set_xticklabels([display_names.get(m, m) for m in models], fontsize=12)
        ax.legend(fontsize=10, ncols=2, loc="upper center")
        ax.set_ylim(0, y_max * 1.35)
        plt.savefig(plot_save_path, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    setup_logging()
    tyro.cli(main)
