"""Streamline and quiver overlays for Eulerian velocity fields."""

import numpy as np
from matplotlib.axes import Axes


def addStreamlines(
    ax: Axes,
    u: np.ndarray,
    v: np.ndarray,
    density: float = 1.2,
    alpha: float = 0.35,
    linewidth: float = 0.6,
    color: str = "#8b949e",
) -> None:
    """
    Draw streamlines on top of an existing axes.

    Parameters
    ----------
    ax:        target matplotlib Axes.
    u:         x-velocity field, shape (height, width).
    v:         y-velocity field, shape (height, width).
    density:   streamplot density parameter.
    alpha:     line alpha.
    linewidth: line width.
    color:     line color.
    """
    h, w = u.shape
    xs = np.linspace(0, w - 1, w)
    ys = np.linspace(0, h - 1, h)
    ax.streamplot(
        xs,
        ys,
        u,
        v,
        density=density,
        linewidth=linewidth,
        color=color,
        arrowsize=0.8,
        zorder=5,
        integration_direction="forward",
    )
    # streamplot sets alpha on the lines collection
    for col in ax.collections:
        col.set_alpha(alpha)


def addVorticityOverlay(
    ax: Axes,
    vorticity: np.ndarray,
    cmap: str = "RdBu_r",
    alpha: float = 0.25,
    vPercentile: float = 99.0,
) -> None:
    """
    Overlay a semi-transparent vorticity field on top of existing content.

    Parameters
    ----------
    ax:          target Axes.
    vorticity:   curl field, shape (height, width).
    cmap:        colormap name.
    alpha:       overlay alpha.
    vPercentile: percentile used for symmetric vmin/vmax.
    """
    vMax = float(np.percentile(np.abs(vorticity), vPercentile))
    ax.imshow(
        vorticity,
        origin="lower",
        cmap=cmap,
        vmin=-vMax,
        vmax=vMax,
        alpha=alpha,
        zorder=4,
        aspect="auto",
    )
