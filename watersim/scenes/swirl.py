"""
Two-source swirl: Stable Fluids on a 128 x 128 grid.

Two counter-rotating dye sources orbit the centre, leaving spiraling streaks.
Vorticity is faded in over the last third of the animation as a soft overlay
so the eye sees the streak picture first, then the underlying rotation field.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import (
    applyDarkTheme, addFooter, PALETTES, FG_PRIMARY, FG_SECONDARY,
)
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


GRID_SIZE = 192
FRAMES = 130
FPS = 30
DENSITY_AMOUNT = 140.0
FORCE_SCALE = 4.0
ORBIT_RADIUS_FRAC = 0.22
OUTPUT = os.path.join(PLOT_DIR, "saved_simulation.gif")

VORT_START_FRAC = 0.65


def _addFluidAtPos(
    solver: StableFluidsSolver,
    x: float, y: float, dx: float, dy: float, radius: int = 4,
) -> None:
    n = solver.size
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di ** 2 + dj ** 2 <= radius ** 2:
                xi = int(np.clip(x + di, 0, n - 1))
                yj = int(np.clip(y + dj, 0, n - 1))
                solver.densityPrev[xi, yj] += DENSITY_AMOUNT
                solver.uPrev[xi, yj] += dx * FORCE_SCALE
                solver.vPrev[xi, yj] += dy * FORCE_SCALE


def runSwirl() -> None:
    """Generate and save the swirling animation."""
    applyDarkTheme()
    ensurePlotDir()

    solver = StableFluidsSolver(size=GRID_SIZE, visc=1e-6)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04)
    addFooter(fig)

    im = ax.imshow(
        solver.density,
        origin="lower",
        cmap=PALETTES["sequential"],
        vmin=0, vmax=1,
        interpolation="bilinear",
    )
    ax.set_title(
        "Two-Source Swirl",
        color=FG_PRIMARY, fontsize=14, pad=18, loc="left", x=0.02,
    )
    ax.text(
        0.02, 1.02,
        "Stable Fluids · counter-rotating dye sources",
        transform=ax.transAxes,
        color=FG_SECONDARY, fontsize=9, ha="left", va="bottom",
    )
    ax.axis("off")

    vortIm = ax.imshow(
        np.zeros((GRID_SIZE, GRID_SIZE)),
        origin="lower",
        cmap=PALETTES["vorticity"],
        alpha=0.0, vmin=-1, vmax=1,
        interpolation="bilinear",
        zorder=4,
    )

    vortStart = int(FRAMES * VORT_START_FRAC)

    def update(frame: int) -> list:
        t = frame / FRAMES
        cx, cy = GRID_SIZE / 2, GRID_SIZE / 2
        r1 = GRID_SIZE * ORBIT_RADIUS_FRAC * (1.0 - t * 0.3)
        r2 = GRID_SIZE * (ORBIT_RADIUS_FRAC - 0.04) * (1.0 - t * 0.3)

        angle1 = 2 * np.pi * t * 2.5
        angle2 = angle1 + np.pi

        x1 = cx + r1 * np.cos(angle1)
        y1 = cy + r1 * np.sin(angle1)
        x2 = cx + r2 * np.cos(angle2)
        y2 = cy + r2 * np.sin(angle2)

        prevAngle1 = 2 * np.pi * (frame - 1) / FRAMES * 2.5
        dx1 = (x1 - (cx + r1 * np.cos(prevAngle1))) * FORCE_SCALE
        dy1 = (y1 - (cy + r1 * np.sin(prevAngle1))) * FORCE_SCALE
        dx2 = -dx1
        dy2 = -dy1

        _addFluidAtPos(solver, x1, y1, dx1, dy1)
        _addFluidAtPos(solver, x2, y2, dx2, dy2)
        solver.step()

        # Mild density decay so spirals don't saturate
        solver.density *= 0.992

        im.set_data(solver.density)
        im.set_clim(0, max(float(solver.density.max()), 1e-6))

        if frame >= vortStart:
            vort = solver.getVorticity()
            vMax = float(np.percentile(np.abs(vort + 1e-9), 99))
            vortIm.set_data(vort)
            vortIm.set_clim(-vMax, vMax)
            alpha = min(0.4, 0.4 * (frame - vortStart) / (FRAMES - vortStart + 1))
            vortIm.set_alpha(alpha)

        if frame % 60 == 0:
            print(f"  Swirl: frame {frame}/{FRAMES}")
        return [im, vortIm]

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT, dpi=92)
