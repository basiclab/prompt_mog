import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import tyro
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText

from eval_utils.vendi import pixel_vendi_score

plt.style.use(["science"])
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "axes.linewidth": 0.8,
        "axes.titleweight": "semibold",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)
rng = np.random.default_rng(10)


def make_cmap(hex_color: str = "#C81E1E") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("white_to_color", ["#fffaf7", hex_color], N=256)


CMAP = make_cmap("#C81E1E")  # crimsonish red
CONTOUR_LINE = "#3a3a3a"


def circle_centers(k: int, r: float = 3.0) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)  # (k, 2)


def gaussian_pdf(xy: np.ndarray, mu: np.ndarray, sig: float) -> np.ndarray:
    cov_inv = np.eye(2) / (sig**2)
    diff = xy - mu
    expo = np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
    norm = 1.0 / (2 * np.pi * (sig**2))
    return norm * np.exp(-0.5 * expo)


def mixture_density_grid(
    centers: np.ndarray, sig: float, weights: np.ndarray | None = None, lim: float = 4.2, res: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if weights is None:
        weights = np.full(centers.shape[0], 1.0 / centers.shape[0])
    xs = np.linspace(-lim, lim, res)
    ys = np.linspace(-lim, lim, res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    XY = np.stack([X, Y], axis=-1)  # (res,res,2)
    Z = np.zeros_like(X)
    for w, mu in zip(weights, centers, strict=True):
        Z += w * gaussian_pdf(XY, mu, sig)
    return xs, ys, Z


def samples_from_mog(centers: np.ndarray, sig: float, total_samples: int = 3000) -> np.ndarray:
    """For each global sample: draw one sample per component, draw Dirichlet weightscombine."""
    k = centers.shape[0]
    choices = rng.integers(0, k, size=total_samples)
    points = centers[choices] + rng.normal(scale=sig, size=(total_samples, 2))
    return points


def prettify_axes(ax: plt.Axes, lim: float = 4.2):
    ax.set_aspect("equal", "box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(
        top=False,
        right=False,
        which="both",
        direction="out",
        length=3,
        width=0.6,
        labelsize=13,
    )
    ax.text(
        0.015,
        0.985,
        "y",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontstyle="italic",
    )
    ax.text(
        0.985,
        0.015,
        "x",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=18,
        fontstyle="italic",
    )


def draw_panel(
    k: int,
    radius: float,
    sigma: float,
    n_samples: int,
    grid_lim: float,
    grid_res: int,
    levels: int,
    scatter_size: float,
    ax: plt.Axes,
    title: str | None = None,
):
    centers = circle_centers(k, r=radius)
    samples = samples_from_mog(centers, sigma, total_samples=n_samples)
    xs, ys, Z = mixture_density_grid(centers, sigma, lim=grid_lim, res=grid_res)

    ax.contourf(xs, ys, Z, levels=levels, cmap=CMAP, antialiased=True)
    ax.scatter(samples[:, 0], samples[:, 1], s=scatter_size, alpha=0.35, ec="none", color="#8B0000")
    prettify_axes(ax, lim=grid_lim)

    vendi_score = pixel_vendi_score(samples)
    at = AnchoredText(
        f" Vendi ({title}): {vendi_score:.2f}",
        prop=dict(size=14.5),
        frameon=True,
        loc="upper right",
        borderpad=0.6,
    )
    at.patch.set_boxstyle("round,pad=0.3,rounding_size=0.8")
    at.patch.set_alpha(0.9)
    ax.add_artist(at)


def main(
    n: int = 2,  # number of modes
    radius: float = 3.0,
    sigma: float = 0.5,
    n_samples: int = 1000,
    grid_lim: float = 4.2,
    grid_res: int = 400,
    levels: int = 22,
    scatter_size: float = 12,
    save_path: str = "assets/toy_example.pdf",
):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.6), constrained_layout=True)
    draw_panel(
        k=n,
        radius=radius,
        sigma=sigma,
        n_samples=n_samples,
        grid_lim=grid_lim,
        grid_res=grid_res,
        levels=levels,
        scatter_size=scatter_size,
        ax=ax1,
        title=f"{n} modes",
    )
    draw_panel(
        k=n + 1,
        radius=radius,
        sigma=sigma,
        n_samples=n_samples,
        grid_lim=grid_lim,
        grid_res=grid_res,
        levels=levels,
        scatter_size=scatter_size,
        ax=ax2,
        title=f"{n + 1} modes",
    )
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    tyro.cli(main)
