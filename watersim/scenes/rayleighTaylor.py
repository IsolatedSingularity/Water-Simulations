"""
Rayleigh-Taylor instability scene.

Dense fluid initialized on top, light fluid on bottom, with a small sinusoidal
perturbation at the interface. Implemented via Stable Fluids with a passive
scalar field tracking the two fluid regions.

Resolution: 192 x 288 grid, 600 frames.
Reference: Sharp 1984, "An overview of Rayleigh-Taylor instability".
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import applyDarkTheme, addFooter, PALETTES
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


WIDTH = 192
HEIGHT = 288
FRAMES = 300
FPS = 30
OUTPUT = os.path.join(PLOT_DIR, "rayleigh_taylor.gif")

# Perturbation at interface
PERT_AMPLITUDE = 0.015   # fraction of HEIGHT
PERT_WAVENUMBER = 3      # number of sinusoidal bumps across width

# Gravity-driven buoyancy: heavy fluid on top gets a downward source term
GRAVITY_STRENGTH = 0.12


def _initScalar(width: int, height: int) -> np.ndarray:
    """
    Return passive scalar: 1.0 = heavy fluid (top half), 0.0 = light fluid (bottom).
    Interface perturbed by a small sinusoid to seed instability.
    """
    scalar = np.zeros((height, width), dtype=np.float64)
    xs = np.arange(width)
    interfaceY = height // 2 + PERT_AMPLITUDE * height * np.sin(
        2 * np.pi * PERT_WAVENUMBER * xs / width
    )
    for col, yInterface in enumerate(interfaceY):
        yCeil = int(np.ceil(yInterface))
        scalar[yCeil:, col] = 1.0   # top (rows > interface) = heavy
    return scalar


class _RTSolver:
    """
    Rayleigh-Taylor solver: Stable Fluids on a HEIGHT x WIDTH grid with
    buoyancy source proportional to the passive scalar (heavy fluid).

    Note: StableFluidsSolver is square; we implement a rectangular variant
    directly here to match the 192 x 288 domain without padding waste.
    """

    DT: float = 0.2
    VISC: float = 1e-4

    def __init__(self, width: int, height: int, scalar0: np.ndarray) -> None:
        self.width = width
        self.height = height
        self.u: np.ndarray = np.zeros((height, width), dtype=np.float64)
        self.v: np.ndarray = np.zeros_like(self.u)
        self.scalar: np.ndarray = scalar0.copy()

    def _advect(
        self, d: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        H, W = self.height, self.width
        dtScale = self.DT
        i = np.arange(1, H - 1)[:, np.newaxis]
        j = np.arange(1, W - 1)[np.newaxis, :]
        x = np.clip(i - dtScale * u[1:-1, 1:-1] * H, 0.5, H - 1.5)
        y = np.clip(j - dtScale * v[1:-1, 1:-1] * W, 0.5, W - 1.5)
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
        # periodic left/right boundaries
        out[:, 0] = out[:, -2]
        out[:, -1] = out[:, 1]
        # no-slip top/bottom
        out[0, :] = 0.0
        out[-1, :] = 0.0
        return out

    def _project(self) -> None:
        H, W = self.height, self.width
        div = np.zeros((H, W), dtype=np.float64)
        p = np.zeros_like(div)
        div[1:-1, 1:-1] = -0.5 * (
            self.u[1:-1, 2:] - self.u[1:-1, :-2]
            + self.v[2:, 1:-1] - self.v[:-2, 1:-1]
        )
        for _ in range(20):
            p[1:-1, 1:-1] = (
                div[1:-1, 1:-1]
                + p[1:-1, 2:] + p[1:-1, :-2]
                + p[2:, 1:-1] + p[:-2, 1:-1]
            ) / 4.0
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, -2]
            p[:, -1] = p[:, 1]
        self.u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
        self.v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
        # periodic left/right
        self.u[:, 0] = self.u[:, -2]
        self.u[:, -1] = self.u[:, 1]

    def step(self) -> None:
        """One RT step: buoyancy source, advect, project, advect scalar."""
        # Buoyancy: heavy fluid (scalar~1) pushed down; light fluid pushed up
        self.v += self.DT * GRAVITY_STRENGTH * (self.scalar - 0.5) * (-1.0)
        self.u = self._advect(self.u, self.u, self.v)
        self.v = self._advect(self.v, self.u, self.v)
        self._project()
        self.scalar = self._advect(self.scalar, self.u, self.v)
        self.scalar = np.clip(self.scalar, 0.0, 1.0)


def runRayleighTaylor() -> None:
    """Generate and save the Rayleigh-Taylor instability animation."""
    applyDarkTheme()
    ensurePlotDir()

    scalar0 = _initScalar(WIDTH, HEIGHT)
    solver = _RTSolver(WIDTH, HEIGHT, scalar0)

    fig, ax = plt.subplots(figsize=(3.4, 5.0))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.04)
    addFooter(fig)

    cmap = PALETTES["diverging"]
    im = ax.imshow(
        solver.scalar,
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="bilinear",
    )
    ax.set_title(
        "Rayleigh-Taylor Instability  ·  Stable Fluids", color="#e6edf3", pad=8
    )
    ax.axis("off")

    def update(frame: int) -> list:
        for _ in range(2):
            solver.step()
        im.set_data(solver.scalar)
        if frame % 60 == 0:
            print(f"  RT: frame {frame}/{FRAMES}")
        return [im]

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT, dpi=80)
