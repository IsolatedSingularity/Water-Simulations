"""
PIC/FLIP dam-break scene.

Wide 192 x 96 basin. A column of fluid on the left collapses under gravity,
hits the right wall, and sloshes back. Particles are drawn larger than in the
previous version so the wave front is clearly readable on a thumbnail-sized
GIF, with a faint motion-blur trail behind each particle.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from watersim.solvers.picFlip import HybridSolver
from watersim.viz.theme import (
    applyDarkTheme, addFooter, PALETTES,
    FG_PRIMARY, FG_SECONDARY, ACCENT_BLUE,
)
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


GRID_WIDTH = 192
GRID_HEIGHT = 96
N_PER_ROW = 50
FRAMES = 340
FPS = 30
OUTPUT = os.path.join(PLOT_DIR, "hybrid_simulation.gif")


def runDamBreak() -> None:
    """Generate and save the PIC/FLIP dam-break animation."""
    applyDarkTheme()
    ensurePlotDir()

    solver = HybridSolver(
        gridWidth=GRID_WIDTH,
        gridHeight=GRID_HEIGHT,
        nParticlesPerRow=N_PER_ROW,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.06)
    addFooter(fig)

    pos = solver.getParticlePositions()
    speeds = solver.getParticleSpeeds() + 1.0

    sc = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=speeds,
        cmap=PALETTES["particle"],
        s=6.0,
        linewidths=0,
        norm=LogNorm(vmin=1.0, vmax=20.0),
        zorder=3,
    )
    ax.set_xlim(0, GRID_WIDTH)
    ax.set_ylim(0, GRID_HEIGHT)
    ax.set_title(
        "PIC/FLIP Dam Break",
        color=FG_PRIMARY, fontsize=14, pad=8, loc="left", x=0.02,
    )
    ax.text(
        0.98, 1.02,
        "Hybrid solver · 192 × 96 basin · α = 0.95",
        transform=ax.transAxes,
        color=FG_SECONDARY, fontsize=9, ha="right", va="bottom",
    )
    ax.axis("off")

    trail: list[np.ndarray] = []
    trailScatters: list = []
    HOLD_FRAMES = 25  # initial pause showing the dam at rest

    def update(frame: int) -> list:
        nonlocal trail, trailScatters

        if frame >= HOLD_FRAMES:
            solver.step()
        pos = solver.getParticlePositions()
        speeds = solver.getParticleSpeeds() + 1.0

        sc.set_offsets(pos)
        sc.set_array(speeds)
        vmax = max(float(np.percentile(speeds, 98)), 2.0)
        sc.norm.vmin = 1.0
        sc.norm.vmax = vmax

        for ts in trailScatters:
            ts.remove()
        trailScatters.clear()

        trail.append(pos.copy())
        if len(trail) > 4:
            trail.pop(0)

        for ti, tPos in enumerate(trail[:-1]):
            alpha = 0.07 * (ti + 1)
            ts = ax.scatter(
                tPos[:, 0], tPos[:, 1],
                c=ACCENT_BLUE, s=4.0,
                linewidths=0, alpha=alpha, zorder=2,
            )
            trailScatters.append(ts)

        if frame % 60 == 0:
            print(f"  DamBreak: frame {frame}/{FRAMES}")
        return [sc] + trailScatters

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT)
