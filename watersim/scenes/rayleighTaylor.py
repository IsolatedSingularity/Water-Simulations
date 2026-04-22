"""
Rayleigh-Taylor instability.

Heavy fluid (top) sits over light fluid (bottom) in a gravitational field.
A multi-mode sinusoidal perturbation seeds the interface; small disturbances
grow exponentially and roll up into the characteristic mushroom-cap spikes.

Implemented as Stable Fluids on a rectangular HEIGHT x WIDTH grid with a
buoyancy source proportional to the passive scalar tracking the dense fluid.
Periodic left/right boundaries; no-slip top and bottom.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from watersim.viz.theme import (
    applyDarkTheme, addFooter, PALETTES, FG_PRIMARY, FG_SECONDARY,
)
from watersim.viz.animator import saveAnimation, ensurePlotDir, PLOT_DIR


WIDTH = 192
HEIGHT = 280
FRAMES = 240
FPS = 30
OUTPUT = os.path.join(PLOT_DIR, "rayleigh_taylor.gif")

# Larger perturbation + several wavenumber modes => irregular interface that
# evolves into proper mushrooms instead of a perfectly periodic stack.
PERT_AMPLITUDE = 0.04
PERT_MODES = (3, 5, 8)
GRAVITY_STRENGTH = 0.35
SUBSTEPS = 2


def _initScalar(width: int, height: int) -> np.ndarray:
    """1.0 = heavy fluid (top), 0.0 = light fluid (bottom)."""
    scalar = np.zeros((height, width), dtype=np.float64)
    xs = np.arange(width)
    rng = np.random.default_rng(seed=7)
    phases = rng.uniform(0, 2 * np.pi, size=len(PERT_MODES))
    amps = np.array([1.0, 0.55, 0.3])
    bumps = np.zeros(width)
    for k, phi, a in zip(PERT_MODES, phases, amps):
        bumps += a * np.sin(2 * np.pi * k * xs / width + phi)
    bumps = bumps / np.max(np.abs(bumps))  # normalise to [-1, 1]
    interfaceY = height // 2 + PERT_AMPLITUDE * height * bumps
    for col, yInterface in enumerate(interfaceY):
        yCeil = int(np.ceil(yInterface))
        scalar[yCeil:, col] = 1.0
    return scalar


class _RTSolver:
    """Stable Fluids on a HEIGHT x WIDTH grid with buoyancy."""

    DT: float = 0.18

    def __init__(self, width: int, height: int, scalar0: np.ndarray) -> None:
        self.width = width
        self.height = height
        self.u: np.ndarray = np.zeros((height, width), dtype=np.float64)
        self.v: np.ndarray = np.zeros_like(self.u)
        self.scalar: np.ndarray = scalar0.copy()

    def _advect(self, d, u, v):
        H, W = self.height, self.width
        dt = self.DT
        i = np.arange(1, H - 1)[:, np.newaxis]
        j = np.arange(1, W - 1)[np.newaxis, :]
        x = np.clip(i - dt * u[1:-1, 1:-1] * H, 0.5, H - 1.5)
        y = np.clip(j - dt * v[1:-1, 1:-1] * W, 0.5, W - 1.5)
        i0 = x.astype(int); i1 = i0 + 1
        j0 = y.astype(int); j1 = j0 + 1
        s1 = x - i0; s0 = 1.0 - s1
        t1 = y - j0; t0 = 1.0 - t1
        out = np.zeros_like(d)
        out[1:-1, 1:-1] = (
            s0 * (t0 * d[i0, j0] + t1 * d[i0, j1])
            + s1 * (t0 * d[i1, j0] + t1 * d[i1, j1])
        )
        out[:, 0] = out[:, -2]
        out[:, -1] = out[:, 1]
        out[0, :] = 0.0
        out[-1, :] = 0.0
        return out

    def _project(self):
        H, W = self.height, self.width
        div = np.zeros((H, W), dtype=np.float64)
        p = np.zeros_like(div)
        div[1:-1, 1:-1] = -0.5 * (
            self.u[1:-1, 2:] - self.u[1:-1, :-2]
            + self.v[2:, 1:-1] - self.v[:-2, 1:-1]
        )
        for _ in range(25):
            p[1:-1, 1:-1] = (
                div[1:-1, 1:-1]
                + p[1:-1, 2:] + p[1:-1, :-2]
                + p[2:, 1:-1] + p[:-2, 1:-1]
            ) / 4.0
            p[0, :] = p[1, :]; p[-1, :] = p[-2, :]
            p[:, 0] = p[:, -2]; p[:, -1] = p[:, 1]
        self.u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
        self.v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
        self.u[:, 0] = self.u[:, -2]
        self.u[:, -1] = self.u[:, 1]

    def step(self) -> None:
        # Buoyancy: heavy fluid (scalar~1) accelerates downward
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

    fig, ax = plt.subplots(figsize=(3.6, 5.2))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.04)
    addFooter(fig)

    im = ax.imshow(
        solver.scalar,
        origin="lower",
        cmap=PALETTES["fluidSplit"],
        vmin=0.0, vmax=1.0,
        aspect="auto",
        interpolation="bilinear",
    )
    ax.set_title(
        "Rayleigh-Taylor",
        color=FG_PRIMARY, fontsize=13, pad=6, loc="left", x=0.02,
    )
    ax.text(
        0.98, 1.02,
        f"{WIDTH}×{HEIGHT} · g={GRAVITY_STRENGTH}",
        transform=ax.transAxes,
        color=FG_SECONDARY, fontsize=8, ha="right", va="bottom",
    )
    ax.axis("off")

    def update(frame: int) -> list:
        for _ in range(SUBSTEPS):
            solver.step()
        im.set_data(solver.scalar)
        if frame % 60 == 0:
            print(f"  RT: frame {frame}/{FRAMES}")
        return [im]

    saveAnimation(fig, update, FRAMES, FPS, OUTPUT, dpi=70)
