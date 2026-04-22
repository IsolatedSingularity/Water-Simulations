"""Dark visual theme and color palette for watersim."""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Colormap
from typing import Any


DARK_RC: dict[str, Any] = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "savefig.facecolor": "#0d1117",
    "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3",
    "axes.edgecolor": "#30363d",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.titlesize": 16,
    "axes.titleweight": "semibold",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

PALETTES: dict[str, Colormap] = {
    "sequential": sns.color_palette("mako", as_cmap=True),
    "diverging": sns.diverging_palette(220, 20, s=80, l=55, as_cmap=True),
    "vorticity": sns.diverging_palette(200, 15, s=85, l=50, as_cmap=True),
    "particle": sns.color_palette("crest", as_cmap=True),
}

FOOTER_TEXT = "watersim · 2026"
FOOTER_COLOR = "#484f58"


def applyDarkTheme() -> None:
    """Apply the dark rcParams globally."""
    plt.rcParams.update(DARK_RC)


def addFooter(fig: "plt.Figure") -> None:
    """Add 'watersim · 2026' footer in bottom-right of figure."""
    fig.text(
        0.98,
        0.01,
        FOOTER_TEXT,
        ha="right",
        va="bottom",
        color=FOOTER_COLOR,
        fontsize=9,
    )
