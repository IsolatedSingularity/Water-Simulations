"""Visual theme and color palettes for watersim.

Tokyo Night-inspired palette: softer than pure black backgrounds, with
two-toned panels (figure-bg slightly darker than axes-bg) for a depth feel
similar to VS Code editor windows. Accent colours are coordinated across all
scenes so the gallery reads as one coherent set of plots rather than six
unrelated screenshots.
"""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palette constants
# ---------------------------------------------------------------------------

# Base surfaces (two-toned, like a VS Code editor inside a darker shell)
BG_FIGURE = "#16161e"  # outer frame  (storm bg)
BG_AXES = "#1a1b26"  # axes panel   (slightly lighter)
BG_PANEL = "#24283b"  # accent panel
GRID_LINE = "#2e3346"  # subtle gridlines / spines

# Foreground typography
FG_PRIMARY = "#c0caf5"  # soft lavender-white (titles, body)
FG_SECONDARY = "#9aa5ce"  # muted lavender (subtitles, axis labels)
FG_TERTIARY = "#565f89"  # comment grey  (footer, annotations)

# Accents (Tokyo Night Storm)
ACCENT_BLUE = "#7aa2f7"
ACCENT_CYAN = "#7dcfff"
ACCENT_PURPLE = "#bb9af7"
ACCENT_MAGENTA = "#f7768e"
ACCENT_ORANGE = "#ff9e64"
ACCENT_GREEN = "#9ece6a"
ACCENT_YELLOW = "#e0af68"


DARK_RC: dict[str, Any] = {
    "figure.facecolor": BG_FIGURE,
    "axes.facecolor": BG_AXES,
    "savefig.facecolor": BG_FIGURE,
    "text.color": FG_PRIMARY,
    "axes.labelcolor": FG_SECONDARY,
    "axes.edgecolor": GRID_LINE,
    "axes.titlecolor": FG_PRIMARY,
    "xtick.color": FG_SECONDARY,
    "ytick.color": FG_SECONDARY,
    "axes.titlesize": 13,
    "axes.titleweight": "regular",
    "axes.labelsize": 10,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID_LINE,
    "grid.alpha": 0.4,
}


def _makeSequentialDeep() -> Colormap:
    """Bg -> blue -> cyan -> soft white. For density / dye fields."""
    return LinearSegmentedColormap.from_list(
        "tnSequential",
        [
            (0.00, BG_AXES),
            (0.20, "#3d59a1"),
            (0.55, ACCENT_BLUE),
            (0.85, ACCENT_CYAN),
            (1.00, "#dfe6f5"),
        ],
        N=256,
    )


def _makeDiverging() -> Colormap:
    """Magenta -> bg -> blue. For divergence and signed scalar fields."""
    return LinearSegmentedColormap.from_list(
        "tnDiverging",
        [
            (0.00, ACCENT_MAGENTA),
            (0.25, "#a85673"),
            (0.50, BG_AXES),
            (0.75, "#5a7ac0"),
            (1.00, ACCENT_BLUE),
        ],
        N=256,
    )


def _makeVorticity() -> Colormap:
    """Orange -> bg -> cyan. Warm/cool split for curl fields."""
    return LinearSegmentedColormap.from_list(
        "tnVorticity",
        [
            (0.00, ACCENT_ORANGE),
            (0.25, "#a8704a"),
            (0.50, BG_AXES),
            (0.75, "#4f8ab0"),
            (1.00, ACCENT_CYAN),
        ],
        N=256,
    )


def _makeParticle() -> Colormap:
    """Bg -> purple -> magenta -> warm white. For particle speed."""
    return LinearSegmentedColormap.from_list(
        "tnParticle",
        [
            (0.00, "#3b3a5a"),
            (0.30, ACCENT_PURPLE),
            (0.65, ACCENT_MAGENTA),
            (1.00, "#fbe8c4"),
        ],
        N=256,
    )


def _makeFluidSplit() -> Colormap:
    """Heavy fluid -> light fluid. Two-tone for Rayleigh-Taylor."""
    return LinearSegmentedColormap.from_list(
        "tnFluidSplit",
        [
            (0.00, ACCENT_CYAN),  # light fluid (bottom in image)
            (0.45, "#4f8ab0"),
            (0.55, "#a8704a"),
            (1.00, ACCENT_MAGENTA),  # heavy fluid (top)
        ],
        N=256,
    )


PALETTES: dict[str, Colormap] = {
    "sequential": _makeSequentialDeep(),
    "diverging": _makeDiverging(),
    "vorticity": _makeVorticity(),
    "particle": _makeParticle(),
    "fluidSplit": _makeFluidSplit(),
}


FOOTER_TEXT = "watersim · 2026"
FOOTER_COLOR = FG_TERTIARY


def applyDarkTheme() -> None:
    """Apply the Tokyo Night-inspired rcParams globally."""
    plt.rcParams.update(DARK_RC)


def addFooter(fig: "plt.Figure") -> None:
    """Add 'watersim · 2026' footer in bottom-right of figure."""
    fig.text(
        0.98,
        0.012,
        FOOTER_TEXT,
        ha="right",
        va="bottom",
        color=FOOTER_COLOR,
        fontsize=8,
        family="DejaVu Sans",
    )
