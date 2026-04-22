"""
Lid-driven cavity scene.

Square cavity (192 x 192). Top wall slides at constant velocity, three other
walls are no-slip. The moving lid drags fluid along, generating a primary
recirculating vortex with secondary corner eddies at high Re.

Single-panel composition: speed magnitude as the colour field with streamlines
overlaid every 15 frames. Reference benchmark: Ghia, Ghia & Shin 1982.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import (
    applyDarkTheme, addFooter, PALETTES, FG_PRIMARY, FG_SECONDARY,
)
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


SIZE = 192
LID_VELOCITY = 1.0
VISCOSITY = 0.005
FRAMES = 200
FPS = 30
OUTPUT = os.path.join(PLOT_DIR, "lid_driven_cavity.gif")

RE = LID_VELOCITY * SIZE / VISCOSITY


def runLidDrivenCavity() -> None:
    """Generate and save the lid-driven cavity animation."""
    applyDarkTheme()
    ensurePlotDir()

    solver = StableFluidsSolver(
        size=SIZE, visc=VISCOSITY, topWallVelocity=LID_VELOCITY,
    )

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.04)
    addFooter(fig)

    speedInit = np.sqrt(solver.u ** 2 + solver.v ** 2)
    im = ax.imshow(
        speedInit,
        origin="lower",
        cmap=PALETTES["sequential"],
        vmin=0, vmax=LID_VELOCITY,
        aspect="equal",
        interpolation="bilinear",
    )

    ax.set_title(
        "Lid-Driven Cavity",
        color=FG_PRIMARY, fontsize=14, pad=8, loc="left", x=0.02,
    )
    ax.text(
        0.98, 1.02,
        f"Re ≈ {RE:.0f} · Stable Fluids · 192² grid",
        transform=ax.transAxes,
        color=FG_SECONDARY, fontsize=9, ha="right", va="bottom",
    )
    ax.axis("off")

    streamLines = None
    streamArrows: list = []
    xs = np.linspace(0, SIZE - 1, SIZE)
    ys = np.linspace(0, SIZE - 1, SIZE)

    def update(frame: int) -> list:
        nonlocal streamLines, streamArrows

        for _ in range(2):
            solver.step()

        speed = np.sqrt(solver.u ** 2 + solver.v ** 2)
        im.set_data(speed)
        im.set_clim(0, max(float(speed.max()), 1e-6))

        # Tear down previous streamlines (lines + arrow patches)
        if streamLines is not None:
            try:
                streamLines.lines.remove()
            except Exception:
                pass
        for arrow in streamArrows:
            try:
                arrow.remove()
            except Exception:
                pass
        streamArrows = []

        try:
            streamLines = ax.streamplot(
                xs, ys,
                solver.u.T, solver.v.T,
                density=1.1, linewidth=0.55,
                color=FG_PRIMARY, arrowsize=0.8, zorder=5,
            )
            streamLines.lines.set_alpha(0.6)
            # Collect any FancyArrowPatch instances added this frame
            for patch in ax.patches:
                if patch not in streamArrows and patch.__class__.__name__ == "FancyArrowPatch":
                    streamArrows.append(patch)
                    patch.set_alpha(0.6)
        except Exception:
            streamLines = None

        if frame % 60 == 0:
            print(f"  LidCavity: frame {frame}/{FRAMES}")
        return [im, streamLines.lines if streamLines else im] + streamArrows

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT, dpi=80)
