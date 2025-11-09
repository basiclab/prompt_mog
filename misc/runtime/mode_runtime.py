import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from pipeline.prompt_mog.regular_simplex import perform_pmog

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science"])
    plt.rcParams["font.family"] = "Times New Roman"
except Exception:
    pass


def main(
    token_size: int = 2175,
    embed_dim: int = 4096,
    num_runs: int = 10,
    modes: list[int] = [1] + list(range(10, 101, 10)),  # noqa: B008
    plot: bool = False,
    plot_save_path: str = "assets/mode_runtime.pdf",
):
    random_prompt_embeds = torch.randn((1, token_size, embed_dim), device="cuda")
    gamma = 0.5
    sigma = 0.5
    generator = torch.Generator(device="cuda").manual_seed(42)

    runtime_results = {}
    print(f"Benchmarking PMOG with different modes (averaging over {num_runs} runs)")
    print("=" * 60)

    for mode in modes:
        print(f"\nMode: {mode}")

        # Warmup run
        print("  Warmup...", end=" ")
        for _ in range(5):
            perform_pmog(random_prompt_embeds, gamma, mode, sigma, 1, generator)
        torch.cuda.synchronize()
        print("done")

        runtimes = []
        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs}...", end=" ")

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            perform_pmog(random_prompt_embeds, gamma, mode, sigma, 1, generator)
            end_event.record()

            torch.cuda.synchronize()

            runtime = start_event.elapsed_time(end_event)
            runtimes.append(runtime)
            print(f"{runtime:.2f}ms")

        avg_runtime = sum(runtimes) / len(runtimes)
        min_runtime = min(runtimes)
        max_runtime = max(runtimes)
        std_runtime = np.std(runtimes)

        runtime_results[mode] = {
            "avg": avg_runtime,
            "min": min_runtime,
            "max": max_runtime,
            "std": std_runtime,
            "all": runtimes,
        }

        print(
            f"  Average: {avg_runtime:.2f}ms, Std: {std_runtime:.2f}ms, Min: {min_runtime:.2f}ms, Max: {max_runtime:.2f}ms"
        )

    print(f"\n\n{'=' * 60}")
    print("RUNTIME SUMMARY")
    print(f"{'=' * 60}\n")
    for mode in modes:
        avg = runtime_results[mode]["avg"]
        std = runtime_results[mode]["std"]
        print(f"Mode {mode:3d}: {avg:7.2f}ms ± {std:5.2f}ms")

    if plot:
        _, ax = plt.subplots(1, 1, figsize=(8, 3))

        modes_list = list(runtime_results.keys())
        avg_times = [runtime_results[m]["avg"] for m in modes_list]
        std_times = [runtime_results[m]["std"] for m in modes_list]

        ax.plot(
            modes_list,
            avg_times,
            marker="o",
            linewidth=2,
            markersize=8,
            color="#0066CC",
            label="Average Runtime",
        )
        ax.fill_between(
            modes_list,
            [avg_times[i] - std_times[i] for i in range(len(modes_list))],
            [avg_times[i] + std_times[i] for i in range(len(modes_list))],
            alpha=0.3,
            color="#4A90E2",
        )

        for i, (mode, avg_time) in enumerate(zip(modes_list, avg_times, strict=True)):
            if i % 2 == 0:
                ax.text(mode, avg_time * 1.005, f"{avg_time:.1f}ms", ha="center", va="bottom", fontsize=10)

        ax.set_xticks(modes_list[::2])
        ax.set_xticklabels([f"$n={mode}$" for mode in modes_list[::2]], fontsize=11)

        ax.set_xlabel("")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(plot_save_path, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    tyro.cli(main)
