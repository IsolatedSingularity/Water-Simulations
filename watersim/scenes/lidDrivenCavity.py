"""
Lid-driven cavity scene.

Square cavity (192 x 192): top wall moves at constant velocity, other walls
no-slip. Uses StableFluidsSolver with topWallVelocity set.
600 frames to show primary vortex + secondary corner vortices.
Reynolds number annotated from lid velocity, cavity size, viscosity.

Reference: Ghia, Ghia, Shin 1982.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import applyDarkTheme, addFooter, PALETTES
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


SIZE = 192
LID_VELOCITY = 1.0
VISCOSITY = 0.005
FRAMES = 600
FPS = 30
OUTPUT = os.path.join(PLOT_DIR, "lid_driven_cavity.gif")

RE = LID_VELOCITY * SIZE / VISCOSITY


def runLidDrivenCavity() -> None:
    """Generate and save the lid-driven cavity animation."""
    applyDarkTheme()
    ensurePlotDir()

    solver = StableFluidsSolver(
        size=SIZE,
        visc=VISCOSITY,
        topWallVelocity=LID_VELOCITY,
    )

    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig, wspace=0.08)
    axVort = fig.add_subplot(gs[0])
    axStream = fig.add_subplot(gs[1])
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.04, right=0.96)
    addFooter(fig)

    vortCmap = PALETTES["vorticity"]
    denseCmap = PALETTES["sequential"]

    vortInit = solver.getVorticity()
    imVort = axVort.imshow(
        vortInit,
        origin="lower",
        cmap=vortCmap,
        vmin=-0.5,
        vmax=0.5,
        aspect="equal",
        interpolation="bilinear",
    )
    axVort.set_title("Vorticity", color="#e6edf3", fontsize=12)
    axVort.axis("off")

    speedInit = np.sqrt(solver.u ** 2 + solver.v ** 2)
    imSpeed = axStream.imshow(
        speedInit,
        origin="lower",
        cmap=denseCmap,
        vmin=0,
        vmax=LID_VELOCITY,
        aspect="equal",
        interpolation="bilinear",
    )
    axStream.set_title("Speed", color="#e6edf3", fontsize=12)
    axStream.axis("off")

    fig.suptitle(
        f"Lid-Driven Cavity  ·  Re = {RE:.0f}",
        color="#e6edf3",
        fontsize=14,
        fontweight="semibold",
    )

    streamArtists: list = []

    xs = np.linspace(0, SIZE - 1, SIZE)
    ys = np.linspace(0, SIZE - 1, SIZE)

    def update(frame: int) -> list:
        nonlocal streamArtists

        for _ in range(2):
            solver.step()

        vort = solver.getVorticity()
        vMax = float(np.percentile(np.abs(vort + 1e-9), 99))
        imVort.set_data(vort)
        imVort.set_clim(-max(vMax, 0.01), max(vMax, 0.01))

        speed = np.sqrt(solver.u ** 2 + solver.v ** 2)
        imSpeed.set_data(speed)
        imSpeed.set_clim(0, max(float(speed.max()), 1e-6))

        # Streamlines on speed panel every 15 frames
        if frame % 15 == 0:
            for artist in streamArtists:
                try:
                    artist.remove()
                except Exception:
                    pass
            streamArtists.clear()
            try:
                sp = axStream.streamplot(
                    xs, ys,
                    solver.u.T, solver.v.T,
                    density=1.0,
                    linewidth=0.6,
                    color="#e6edf3",
                    arrowsize=0.7,
                    zorder=5,
                )
                for col in axStream.collections[-2:]:
                    col.set_alpha(0.5)
                    streamArtists.append(col)
            except Exception:
                pass

        if frame % 60 == 0:
            print(f"  LidCavity: frame {frame}/{FRAMES}")

        return [imVort, imSpeed] + streamArtists

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT)
