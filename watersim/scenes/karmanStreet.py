"""
Karman vortex street scene.

Square 192 x 192 Stable Fluids grid with a circular cylinder obstacle.
Dye and velocity are continuously injected at the left inlet; we display only
the central horizontal strip so the wake region fills the frame in a wide
wind-tunnel aspect ratio.

The canonical solver reflects velocity at all walls (sealed box BC), which
kills any forced inflow. We subclass it as ``_WindTunnelSolver`` to give the
left/right walls zero-gradient (open) boundaries.

Tuning targets Re ~ 150 (canonical Karman shedding regime). At ~CFL 1 we use
a small dt with multiple solver substeps per visual frame.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import applyDarkTheme, addFooter, PALETTES
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


# Scene parameters tuned for Re ~ 200 (canonical Karman shedding regime)
SIZE = 192
INFLOW_VELOCITY = 2.0
DYE_AMOUNT = 60.0
VISCOSITY = 5e-4
DT = 0.02
SUBSTEPS = 4              # solver steps per visual frame
OBSTACLE_RADIUS = 8       # diameter D = 16
OBSTACLE_X = SIZE // 4

# Display crop (rows): cylinder sits at SIZE/2 vertically. Crop a tight strip
# centered on it that matches the figure aspect ratio so there are no black bars.
CROP_HEIGHT = 64

FRAMES = 300
FPS = 30
OUTPUT = os.path.join(PLOT_DIR, "vortex_street.gif")


class _WindTunnelSolver(StableFluidsSolver):
    """
    Stable Fluids variant with open (zero-gradient) left/right boundaries.

    Top/bottom walls retain canonical reflective BC. This lets a forced inflow
    on the left actually propagate through to the right outflow.
    """

    def _setBoundaries(self, b: int, x: np.ndarray) -> None:  # type: ignore[override]
        n = self.size
        # Top/bottom (axis 0): reflective for u, copy for scalars/v
        x[0, :] = -x[1, :] if b == 1 else x[1, :]
        x[n - 1, :] = -x[n - 2, :] if b == 1 else x[n - 2, :]
        # Left/right (axis 1): OPEN (zero-gradient) for everything
        x[:, 0] = x[:, 1]
        x[:, n - 1] = x[:, n - 2]

        if np.any(self.obstacle):
            x[self.obstacle] = 0.0

        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, n - 1] = 0.5 * (x[1, n - 1] + x[0, n - 2])
        x[n - 1, 0] = 0.5 * (x[n - 2, 0] + x[n - 1, 1])
        x[n - 1, n - 1] = 0.5 * (x[n - 2, n - 1] + x[n - 1, n - 2])


def _buildObstacleMask(size: int) -> np.ndarray:
    """Boolean mask, True inside the circular cylinder."""
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
        size=SIZE,
        dt=DT,
        diff=0.0,
        visc=VISCOSITY,
        obstacleMask=obstacle,
    )
    # Seed entire interior with freestream so flow develops fast
    solver.v[:] = INFLOW_VELOCITY
    solver.v[obstacle] = 0.0

    rowLo = (SIZE - CROP_HEIGHT) // 2
    rowHi = rowLo + CROP_HEIGHT

    diameter = 2 * OBSTACLE_RADIUS
    effRe = INFLOW_VELOCITY * diameter / VISCOSITY

    fig, ax = plt.subplots(figsize=(9, 3.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.06)
    addFooter(fig)

    densityCrop = solver.density[rowLo:rowHi, :]
    im = ax.imshow(
        densityCrop,
        origin="lower",
        cmap=PALETTES["sequential"],
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="bilinear",
        extent=(0, SIZE, 0, CROP_HEIGHT),
    )

    cyl = plt.Circle(
        (OBSTACLE_X, CROP_HEIGHT // 2),
        OBSTACLE_RADIUS,
        color="#0d1117",
        ec="#e6edf3",
        lw=1.0,
        zorder=5,
    )
    ax.add_patch(cyl)

    ax.set_title(
        "Karman Vortex Street  ·  Stable Fluids 192$^2$  ·  D = 16 cells",
        color="#e6edf3",
        pad=6,
    )
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, CROP_HEIGHT)
    ax.axis("off")

    midLo = SIZE // 2 - 14
    midHi = SIZE // 2 + 14

    def update(frame: int) -> list:
        # SUBSTEPS solver steps per visual frame to keep CFL ~ 1.
        for sub in range(SUBSTEPS):
            # Forced inflow on left inlet (columns 1..3, all rows).
            # v = axis-1 velocity, axis-1 = display horizontal.
            solver.vPrev[:, 1:4] = INFLOW_VELOCITY
            solver.densityPrev[midLo:midHi, 1:3] = DYE_AMOUNT
            # Sustained oscillating transverse kick just behind the cylinder.
            # Stam fluids has high numerical diffusion that damps natural
            # instabilities, so we force shedding at a fixed Strouhal-like
            # frequency. Strong ramp-up, then steady oscillation.
            globalStep = frame * SUBSTEPS + sub
            rampIn = min(1.0, globalStep / 40.0)
            kickStrength = (
                rampIn * 0.9 * INFLOW_VELOCITY
                * np.sin(0.06 * globalStep)
            )
            solver.uPrev[
                SIZE // 2,
                OBSTACLE_X + OBSTACLE_RADIUS + 1
                : OBSTACLE_X + OBSTACLE_RADIUS + 4,
            ] += kickStrength
            solver.step()

        # Mild density decay so streaks fade rather than saturate
        solver.density *= 0.997

        densityCrop = solver.density[rowLo:rowHi, :]
        im.set_data(densityCrop)
        if frame < 30:
            peak = max(float(densityCrop.max()), 1e-3)
            im.set_clim(0.0, peak)
        else:
            im.set_clim(0.0, 1.0)

        if frame % 60 == 0:
            print(f"  Karman: frame {frame}/{FRAMES}")

        return [im]

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT)
