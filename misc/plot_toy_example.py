import matplotlib.pyplot as plt
import numpy as np
import tyro

from eval_utils.vendi import score_K

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science"])
    plt.rcParams["font.family"] = "Times New Roman"
except Exception:
    pass

plt.rcParams.update(
    {
        "axes.linewidth": 0.8,
        "axes.titleweight": "semibold",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)

rng = np.random.default_rng(0)


def gaussian_pdf_1d(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def mog_pdf_1d(x: np.ndarray, mus: np.ndarray, sigma: float) -> np.ndarray:
    k = len(mus)
    p = np.zeros_like(x, dtype=float)
    for m in mus:
        p += gaussian_pdf_1d(x, m, sigma)
    return p / k  # uniform weights


def numeric_entropy_1d(x: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-300, None)
    return float(-np.trapezoid(p * np.log(p), x))


def component_means_1d(n: int, delta: float) -> np.ndarray:
    start = -delta * (n - 1) / 2.0
    return np.array([start + i * delta for i in range(n)], dtype=float)


def sample_mog_1d(mus: np.ndarray, sigma: float, size: int) -> np.ndarray:
    k = len(mus)
    choices = rng.integers(0, k, size=size)
    return mus[choices] + rng.normal(scale=sigma, size=size)


def prettify_axes(
    ax: plt.Axes,
    remove_spines: list[str] = ["top", "right"],  # noqa: B006
):
    for spine in remove_spines:
        ax.spines[spine].set_visible(False)
    ax.tick_params(
        top="top" not in remove_spines,
        right="right" not in remove_spines,
        which="both",
        direction="out",
        length=3,
        width=0.6,
        labelsize=12.5,
    )


def main(
    sigma: float = 0.5,
    delta_mult: float = 6.0,  # separation in units of sigma (≈ disjoint support)
    viz_ns: tuple[int, ...] = (1, 2, 4, 8),
    ns: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 10, 12, 16),
    sample_size: int = 300,  # number of samples per n to compute Vendi
    grid_dx: float = 1e-3,
    margin_sigmas: float = 6.0,
    figsize: tuple[float, float] = (7.0, 4.5),
    save_path: str = "assets/toy_example.pdf",
):
    delta = delta_mult * sigma

    n_max = max(viz_ns)
    mus_max = component_means_1d(n_max, delta)
    lo = mus_max.min() - margin_sigmas * sigma
    hi = mus_max.max() + margin_sigmas * sigma
    x = np.arange(lo, hi, grid_dx)

    # Figure with two vertical subplots (no titles/axis labels on top panel)
    fig, (ax_pdf, ax_bottom) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

    # --- Top: mixture PDFs for selected n ---
    for n in viz_ns:
        mus = component_means_1d(n, delta)
        p = mog_pdf_1d(x, mus, sigma)
        ax_pdf.plot(x, p, label=f"n={n}", linewidth=1.6)
    ax_pdf.legend(frameon=False, fontsize=11.5, loc="upper right")
    prettify_axes(ax_pdf)

    # --- Bottom: entropy vs n (left y) and vendi vs n (right y) ---
    h_single = 0.5 * np.log(2 * np.pi * np.e * sigma**2)  # 1D Gaussian entropy
    H_numeric, H_theory, vendi_vals = [], [], []

    max_row_score = None
    for n in ns[::-1]:
        mus = component_means_1d(n, delta)
        lo_n = mus.min() - margin_sigmas * sigma
        hi_n = mus.max() + margin_sigmas * sigma
        x_n = np.arange(lo_n, hi_n, grid_dx)
        p_n = mog_pdf_1d(x_n, mus, sigma)
        Hn = numeric_entropy_1d(x_n, p_n)
        H_numeric.append(Hn)
        H_theory.append(h_single + np.log(n))

        xs = sample_mog_1d(mus, sigma, sample_size)
        xs = xs.reshape(-1, 1)
        sim = np.abs(xs - xs.T)
        if max_row_score is None:
            max_row_score = np.amax(sim, axis=1, keepdims=True)
        vendi = float(score_K(sim / max_row_score, normalize=False))
        vendi_vals.append(vendi)

    H_numeric = H_numeric[::-1]
    H_theory = H_theory[::-1]
    vendi_vals = vendi_vals[::-1]

    axH = ax_bottom
    ln1 = axH.plot(ns, H_numeric, marker="o", linewidth=1.6, label=r"Estimated $H_n$")
    ln2 = axH.plot(ns, H_theory, marker="o", linestyle="--", linewidth=1.2, label=r"$h+\log n$")
    axH.set_xticks([0] + list(ns))
    axH.set_xticklabels([r"$n=$"] + [str(v) for v in ns], fontsize=12.5)
    axH.set_xlim(0, max(ns) + 1)

    axV = axH.twinx()
    vendi_clean = [np.nan if v is None else v for v in vendi_vals]
    ln3 = axV.plot(ns, vendi_clean, marker="s", linewidth=1.4, label="Vendi Score (VS)", color="orange")

    handles = ln1 + ln2 + ln3
    labels = [h.get_label() for h in handles]
    axH.legend(handles, labels, frameon=False, fontsize=11.5, loc="lower right")

    prettify_axes(axH)
    prettify_axes(axV, remove_spines=["top"])
    axH.text(
        0.0, 1.02, r"$(H_n)$", transform=axH.transAxes, ha="left", va="bottom", fontsize=12.5, clip_on=False
    )
    axH.text(
        1.0, 1.02, r"$(VS)$", transform=axH.transAxes, ha="right", va="bottom", fontsize=12.5, clip_on=False
    )

    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    for n, Hn, Ht, v in zip(ns, H_numeric, H_theory, vendi_vals, strict=True):
        print(f"n={n:>2d} | H_n={Hn:.3f} | h+log n={Ht:.3f} | Vendi={v if v is not None else 'N/A'}")


if __name__ == "__main__":
    tyro.cli(main)
