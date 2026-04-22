"""Shared FuncAnimation wrapper with progress reporting and GIF saving."""

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from typing import Callable


PLOT_DIR = "Plots"


def ensurePlotDir() -> str:
    """Create Plots/ directory if missing and return its path."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    return PLOT_DIR


def saveAnimation(
    fig: Figure,
    updateFn: Callable[[int], list],
    frames: int,
    fps: int,
    outputPath: str,
    interval: int = 33,
) -> None:
    """
    Run a FuncAnimation and save it as a GIF.

    Parameters
    ----------
    fig:        figure to animate.
    updateFn:   frame update callback returning a list of artists.
    frames:     total frame count.
    fps:        frames per second for output GIF.
    outputPath: destination file path (must end in .gif).
    interval:   delay between frames in milliseconds (controls preview speed).
    """
    ensurePlotDir()
    anim = FuncAnimation(
        fig,
        updateFn,
        frames=frames,
        interval=interval,
        blit=True,
    )
    writer = PillowWriter(fps=fps)
    print(f"Saving {frames} frames -> {outputPath}")
    anim.save(outputPath, writer=writer, dpi=120)
    print(f"Saved: {outputPath}")
    plt.close(fig)
