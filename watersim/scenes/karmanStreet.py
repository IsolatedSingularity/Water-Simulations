"""
Karman vortex street scene.

Square 192 x 192 Stable Fluids grid with a circular cylinder obstacle.
We display a wide horizontal strip centered on the cylinder. The visualised
field is the **dye density modulated by the sign of local vorticity**, so the
dye shows where the fluid actually is (streak-line picture, like a wind
tunnel) while the colour tells you which way it is spinning. Combined with
the open left/right boundaries on a Stam fluids subclass and a sustained
oscillating transverse kick behind the cylinder, this produces a clear,
periodic Karman vortex street.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import (
    applyDarkTheme,
    addFooter,
    PALETTES,
    FG_PRIMARY,
    FG_SECONDARY,
)
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


# Scene parameters tuned for visible periodic shedding
SIZE = 256
INFLOW_VELOCITY = 2.0
DYE_AMOUNT = 80.0
VISCOSITY = 5e-4
DT = 0.02
SUBSTEPS = 4
OBSTACLE_RADIUS = 16          # diameter D = 32 cells (visible from a distance)
OBSTACLE_X = SIZE // 4

# Display crop: tight band around cylinder for wind-tunnel aspect
CROP_HEIGHT = 108

FRAMES = 200
FPS = 30
OUTPUT = os.path.join(PLOT_DIR, "vortex_street.gif")


class _WindTunnelSolver(StableFluidsSolver):
    """Stable Fluids variant with open (zero-gradient) left/right boundaries."""

    def _setBoundaries(self, b: int, x: np.ndarray) -> None:  # type: ignore[override]
        n = self.size
        x[0, :] = -x[1, :] if b == 1 else x[1, :]
        x[n - 1, :] = -x[n - 2, :] if b == 1 else x[n - 2, :]
        x[:, 0] = x[:, 1]
        x[:, n - 1] = x[:, n - 2]
        if np.any(self.obstacle):
            x[self.obstacle] = 0.0
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, n - 1] = 0.5 * (x[1, n - 1] + x[0, n - 2])
        x[n - 1, 0] = 0.5 * (x[n - 2, 0] + x[n - 1, 1])
        x[n - 1, n - 1] = 0.5 * (x[n - 2, n - 1] + x[n - 1, n - 2])


def _buildObstacleMask(size: int) -> np.ndarray:
    cx = OBSTACLE_X
    cy = size // 2
    ys = np.arange(size)[:, np.newaxis]
    xs = np.arange(size)[np.newaxis, :]
    return (xs - cx) ** 2 + (ys - cy) ** 2 < OBSTACLE_RADIUS ** 2


def runKarmanStreet() -> None:
    """Generate and save the Karman vortex street animation."""
    applyDarkTheme()
    ensurePlotDir()

    obstacle = _buildObstacleMask(SIZE)
    solver = _WindTunnelSolver(
        size=SIZE, dt=DT, diff=0.0, visc=VISCOSITY, obstacleMask=obstacle,
    )
    solver.v[:] = INFLOW_VELOCITY
    solver.v[obstacle] = 0.0

    rowLo = (SIZE - CROP_HEIGHT) // 2
    rowHi = rowLo + CROP_HEIGHT

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.06)
    addFooter(fig)

    # Initial blank field; we show density * sign(vorticity).
    initial = np.zeros((CROP_HEIGHT, SIZE))
    im = ax.imshow(
        initial,
        origin="lower",
        cmap=PALETTES["vorticity"],
        vmin=-1.0, vmax=1.0,
        aspect="equal",
        interpolation="bilinear",
        extent=(0, SIZE, 0, CROP_HEIGHT),
    )

    cyl = plt.Circle(
        (OBSTACLE_X, CROP_HEIGHT // 2),
        OBSTACLE_RADIUS,
        facecolor="#16161e", edgecolor=FG_SECONDARY, lw=1.2, zorder=5,
    )
    ax.add_patch(cyl)

    ax.set_title(
        "Kármán Vortex Street",
        color=FG_PRIMARY, fontsize=14, pad=8, loc="left", x=0.02,
    )
    ax.text(
        0.98, 1.02,
        f"Stable Fluids · 192² grid · D = {2 * OBSTACLE_RADIUS} cells",
        transform=ax.transAxes,
        color=FG_SECONDARY, fontsize=9, ha="right", va="bottom",
    )
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, CROP_HEIGHT)
    ax.axis("off")

    # Dye injection band centered on cylinder height
    midLo = SIZE // 2 - 18
    midHi = SIZE // 2 + 18

    def update(frame: int) -> list:
        for sub in range(SUBSTEPS):
            solver.vPrev[:, 1:4] = INFLOW_VELOCITY
            solver.densityPrev[midLo:midHi, 1:3] = DYE_AMOUNT
            globalStep = frame * SUBSTEPS + sub
            rampIn = min(1.0, globalStep / 40.0)
            kickStrength = (
                rampIn * 0.9 * INFLOW_VELOCITY * np.sin(0.06 * globalStep)
            )
            solver.uPrev[
                SIZE // 2,
                OBSTACLE_X + OBSTACLE_RADIUS + 1
                : OBSTACLE_X + OBSTACLE_RADIUS + 4,
            ] += kickStrength
            solver.step()

        solver.density *= 0.995

        # Composite field: dye intensity carries the sign of local vorticity.
        densityCrop = solver.density[rowLo:rowHi, :]
        vortCrop = solver.getVorticity()[rowLo:rowHi, :]
        # Normalise dye to [0, 1] then multiply by tanh-compressed vorticity sign.
        dyeNorm = np.clip(densityCrop / max(densityCrop.max(), 1e-3), 0.0, 1.0)
        composite = dyeNorm * np.tanh(8.0 * vortCrop)
        im.set_data(composite)
        im.set_clim(-0.9, 0.9)

        if frame % 60 == 0:
            print(f"  Karman: frame {frame}/{FRAMES}")
        return [im]

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT, dpi=130)
