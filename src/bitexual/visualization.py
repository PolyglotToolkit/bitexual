# type: ignore
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from adiumentum.timestamps import insert_timestamp

from .types import Scores


def plot_scores(
    scores: Scores, len_a: int, len_b: int, location: Path = Path("scores.png")
) -> Path:
    location = insert_timestamp(location)
    alignment = np.zeros((len_a, len_b))
    for (i, j), sc in scores.items():
        alignment[i, j] = sc
    _, ax = plt.subplots(figsize=(8, 6))

    ax.imshow(alignment, aspect="auto", cmap="gray")

    plt.title("Heatmap")
    plt.tight_layout()
    plt.savefig(location, dpi=150, bbox_inches="tight")
    plt.close()

    return location


def plot_alignment(matrix: np.ndarray, location: Path = Path("alignment.png")) -> Path:
    """
    Plot a matrix containing only -1, 0, 1, 2 as a discrete heatmap.

    Args:
        matrix: 2D numpy array with values in {-1, 0, 1, 2}
        ax:     Optional matplotlib Axes to draw on
        labels: Optional dict mapping {-1: "label", 0: "label", 1: "label", 2: "label"}
    """
    location = insert_timestamp(location)
    _, ax = plt.subplots(figsize=(8, 6))

    cmap = ListedColormap(["#241e1e", "#796868", "#285E57", "#B8C10E"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], ncolors=4)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2])
    cbar.ax.set_yticklabels(("-1", "0", "1", "2"))

    plt.title("Ternary heatmap")
    plt.tight_layout()
    plt.savefig(location, dpi=150, bbox_inches="tight")
    plt.close()

    return location
