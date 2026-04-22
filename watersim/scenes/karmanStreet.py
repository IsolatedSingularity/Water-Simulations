"""
Kármán vortex street scene.

Stable Fluids solver on a 256 x 64 grid with a circular obstacle.
500 frames for multiple full shedding periods.
Streamline overlay computed every 10 frames.
Reynolds number annotation from inflow velocity, cylinder diameter, viscosity.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import applyDarkTheme, addFooter, PALETTES
from watersim.viz.overlays import addStreamlines
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


# Scene parameters
WIDTH = 256
HEIGHT = 64
INFLOW_VELOCITY = 1.0
VISCOSITY = 0.0002
OBSTACLE_RADIUS_FRAC = 0.10   # fraction of HEIGHT
FRAMES = 500
FPS = 30
OUTPUT = os.path.join(PLOT_DIR, "vortex_street.gif")

# Reynolds number: Re = U * D / nu  (D = diameter in cell units)
_D = 2 * OBSTACLE_RADIUS_FRAC * HEIGHT
RE = INFLOW_VELOCITY * _D / VISCOSITY


def _buildObstacleMask(width: int, height: int) -> np.ndarray:
    """Return boolean mask with True inside circular cylinder."""
    cx = width // 5
    cy = height // 2
    r = OBSTACLE_RADIUS_FRAC * height
    ys = np.arange(height)[:, np.newaxis]
    xs = np.arange(width)[np.newaxis, :]
    return (xs - cx) ** 2 + (ys - cy) ** 2 < r ** 2


def runKarmanStreet() -> None:
    """Generate and save the Kármán vortex street animation."""
    applyDarkTheme()
    ensurePlotDir()

    obstacle = _buildObstacleMask(WIDTH, HEIGHT)
    solver = StableFluidsSolver(
        size=WIDTH,   # note: solver is square; we crop display
        visc=VISCOSITY,
        obstacleMask=np.zeros((WIDTH, WIDTH), dtype=bool),  # see below
    )
    # Override: build a non-square solver manually using arrays
    # StableFluidsSolver is square; for 256x64 we use only the sub-slice.
    # Simpler approach: use a dedicated rectangular layout.
    solver = _RectangularKarmanSolver(WIDTH, HEIGHT, VISCOSITY, obstacle)

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)
    addFooter(fig)

    vortInit = solver.getVorticity()
    vMax = max(float(np.percentile(np.abs(vortInit + 1e-9), 99)), 0.05)
    im = ax.imshow(
        vortInit,
        origin="lower",
        cmap=PALETTES["vorticity"],
        vmin=-vMax,
        vmax=vMax,
        aspect="auto",
        interpolation="bilinear",
    )
    # Shade obstacle
    ax.imshow(
        obstacle.astype(float),
        origin="lower",
        cmap="gray",
        alpha=0.6,
        aspect="auto",
        zorder=3,
    )
    ax.set_title(
        f"Kármán Vortex Street  ·  Re = {RE:.0f}",
        color="#e6edf3",
        pad=8,
    )
    ax.axis("off")

    scaleText = ax.text(
        WIDTH - 5, 3, f"Re = {RE:.0f}", color="#8b949e", fontsize=9, ha="right"
    )
    streamArtists: list = []

    def update(frame: int) -> list:
        nonlocal vMax, streamArtists

        for _ in range(2):
            solver.step()

        vort = solver.getVorticity()
        vMax99 = float(np.percentile(np.abs(vort + 1e-9), 99))
        vMax = max(vMax99, 0.01)
        im.set_data(vort)
        im.set_clim(-vMax, vMax)

        # Streamline overlay every 10 frames (clear previous)
        if frame % 10 == 0:
            for artist in streamArtists:
                try:
                    artist.remove()
                except Exception:
                    pass
            streamArtists.clear()
            try:
                sp = ax.streamplot(
                    np.arange(WIDTH),
                    np.arange(HEIGHT),
                    solver.u[:HEIGHT, :WIDTH].T,
                    solver.v[:HEIGHT, :WIDTH].T,
                    density=0.8,
                    linewidth=0.5,
                    color="#8b949e",
                    arrowsize=0.6,
                    zorder=5,
                )
                for col in ax.collections[-2:]:
                    col.set_alpha(0.3)
                    streamArtists.append(col)
            except Exception:
                pass

        if frame % 50 == 0:
            print(f"  Kármán: frame {frame}/{FRAMES}")

        return [im, scaleText]

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT)


# ---------------------------------------------------------------------------
# Rectangular solver (StableFluidsSolver adapted for WIDTH x HEIGHT grid)
# ---------------------------------------------------------------------------

class _RectangularKarmanSolver:
    """
    Minimal Stable Fluids implementation for a rectangular (W x H) domain.

    Keeps the obstacle mask and inflow boundary as in the original
    advanced_simulations.py VortexSolver, refactored to camelCase.
    """

    INFLOW: float = INFLOW_VELOCITY
    VISC: float = VISCOSITY
    DT: float = 1.0

    def __init__(
        self, width: int, height: int, visc: float, obstacle: np.ndarray
    ) -> None:
        self.width = width
        self.height = height
        self.visc = visc
        self.obstacle = obstacle

        self.u: np.ndarray = np.zeros((height, width), dtype=np.float64)
        self.v: np.ndarray = np.zeros_like(self.u)
        self.density: np.ndarray = np.zeros_like(self.u)

        # Inflow: left column moves right
        self.u[:, 0] = self.INFLOW

    def _advect(self, d: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Semi-Lagrangian advection."""
        H, W = self.height, self.width
        i = np.arange(1, H - 1)[:, np.newaxis]
        j = np.arange(1, W - 1)[np.newaxis, :]
        x = np.clip(i - u[1:-1, 1:-1], 0.5, H - 1.5)
        y = np.clip(j - v[1:-1, 1:-1], 0.5, W - 1.5)
        i0 = x.astype(int)
        i1 = i0 + 1
        j0 = y.astype(int)
        j1 = j0 + 1
        s1 = x - i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1
        out = np.zeros_like(d)
        out[1:-1, 1:-1] = (
            s0 * (t0 * d[i0, j0] + t1 * d[i0, j1])
            + s1 * (t0 * d[i1, j0] + t1 * d[i1, j1])
        )
        out[self.obstacle] = 0.0
        return out

    def _diffuse(self, d: np.ndarray) -> np.ndarray:
        """5-iteration Jacobi diffusion."""
        a = self.DT * self.visc * self.height * self.width
        out = d.copy()
        for _ in range(5):
            out[1:-1, 1:-1] = (
                d[1:-1, 1:-1]
                + a * (out[:-2, 1:-1] + out[2:, 1:-1] + out[1:-1, :-2] + out[1:-1, 2:])
            ) / (1.0 + 4.0 * a)
            out[self.obstacle] = 0.0
        return out

    def _project(self) -> None:
        """30-iteration Jacobi pressure projection."""
        H, W = self.height, self.width
        div = np.zeros((H, W), dtype=np.float64)
        p = np.zeros_like(div)
        div[1:-1, 1:-1] = -0.5 * (
            self.u[1:-1, 2:] - self.u[1:-1, :-2]
            + self.v[2:, 1:-1] - self.v[:-2, 1:-1]
        )
        div[self.obstacle] = 0.0
        for _ in range(30):
            p[1:-1, 1:-1] = (
                div[1:-1, 1:-1]
                + p[1:-1, 2:] + p[1:-1, :-2]
                + p[2:, 1:-1] + p[:-2, 1:-1]
            ) / 4.0
            p[self.obstacle] = 0.0
            p[0, :] = p[1, :]; p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]
        self.u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
        self.v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
        self.u[self.obstacle] = 0.0
        self.v[self.obstacle] = 0.0
        self.u[:, 0] = self.INFLOW

    def step(self) -> None:
        """One Kármán step: advect, diffuse, project."""
        self.u = self._advect(self.u, self.u, self.v)
        self.v = self._advect(self.v, self.u, self.v)
        self.u = self._diffuse(self.u)
        self.v = self._diffuse(self.v)
        self._project()

    def getVorticity(self) -> np.ndarray:
        """Return curl field."""
        vort = np.zeros_like(self.u)
        vort[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) * 0.5
            - (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) * 0.5
        )
        return vort
