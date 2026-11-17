"""One plot style for all notebooks."""

import matplotlib.pyplot as plt

PALETTE = ["#1f4e79", "#c0504d", "#4f8f4f", "#d08a1f", "#6b4c9a", "#2a9d8f", "#7f7f7f", "#b5651d"]


def apply_style():
    plt.rcParams.update(
        {
            "figure.figsize": (7.5, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )
