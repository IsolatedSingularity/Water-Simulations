"""
PIC/FLIP dam-break scene.

Wide 192 x 96 basin. Particles colored by log-stretched speed.
500 frames for full settle and secondary slosh.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from watersim.solvers.picFlip import HybridSolver
from watersim.viz.theme import applyDarkTheme, addFooter, PALETTES
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


GRID_WIDTH = 192
GRID_HEIGHT = 96
N_PER_ROW = 50
FRAMES = 320
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

    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.06)
    addFooter(fig)

    pos = solver.getParticlePositions()
    speeds = solver.getParticleSpeeds() + 1.0

    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=speeds,
        cmap=PALETTES["particle"],
        s=1.5,
        linewidths=0,
        norm=LogNorm(vmin=1.0, vmax=20.0),
    )
    ax.set_xlim(0, GRID_WIDTH)
    ax.set_ylim(0, GRID_HEIGHT)
    ax.set_title("PIC/FLIP Dam Break  ·  192 x 96 basin", color="#e6edf3", pad=8)
    ax.axis("off")

    # Trail arrays: store last 3 positions for motion blur
    trail: list[np.ndarray] = []
    trailScatters: list = []

    def update(frame: int) -> list:
        nonlocal trail, trailScatters

        solver.step()

        pos = solver.getParticlePositions()
        speeds = solver.getParticleSpeeds() + 1.0

        # Update main scatter
        sc.set_offsets(pos)
        sc.set_array(speeds)
        vmax = max(float(np.percentile(speeds, 98)), 2.0)
        sc.norm.vmin = 1.0
        sc.norm.vmax = vmax

        # Motion-blur trail: render last 3 positions at decaying alpha
        for ts in trailScatters:
            ts.remove()
        trailScatters.clear()

        trail.append(pos.copy())
        if len(trail) > 3:
            trail.pop(0)

        for ti, tPos in enumerate(trail[:-1]):
            alpha = 0.08 * (ti + 1)
            ts = ax.scatter(
                tPos[:, 0], tPos[:, 1],
                c="#4d9de0", s=1.0, linewidths=0, alpha=alpha, zorder=2,
            )
            trailScatters.append(ts)

        if frame % 50 == 0:
            print(f"  DamBreak: frame {frame}/{FRAMES}")

        return [sc] + trailScatters

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT)
