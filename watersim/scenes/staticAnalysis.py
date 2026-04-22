"""
Static comparison: SPH vs Stable Fluids snapshots.

4 fields x 2 solvers laid out as 4 rows x 2 columns. Each row is a single
field (density, pressure, divergence, speed) so the eye compares the two
solvers directly. Bigger panels and per-row colorbars make the figure
readable at gallery thumbnail size.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from watersim.solvers.sph import SPHSolver
from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import (
    applyDarkTheme, addFooter, PALETTES, FG_PRIMARY, FG_SECONDARY,
)
from watersim.viz.animator import ensurePlotDir, PLOT_DIR


SIMULATION_STEPS = 50
N_PER_ROW = 30   # ~1800 particles
OUTPUT = os.path.join(PLOT_DIR, "static_comparison.png")
FIELDS = ("density", "pressure", "divergence", "speed")
FIELD_LABELS = ("Density", "Pressure", "Divergence", "Speed")


def runStaticAnalysis() -> None:
    """Run both solvers for SIMULATION_STEPS and save a 4x2 comparison PNG."""
    applyDarkTheme()
    ensurePlotDir()

    sphSolver = SPHSolver(nParticlesPerRow=N_PER_ROW)
    for _ in range(SIMULATION_STEPS):
        sphSolver.step()
    sphData = sphSolver.getGriddedData()

    sfSolver = StableFluidsSolver(size=64)
    sfSolver.density[:, : sfSolver.size // 3] = 1.0
    # Body force: gravity-equivalent push so the dam actually moves and the
    # comparison shows non-trivial pressure / divergence / speed fields
    # instead of an empty grid.
    for _ in range(SIMULATION_STEPS):
        sfSolver.uPrev[:, :] = -0.15        # downward (axis 0)
        sfSolver.step()
    sfData = sfSolver.getGriddedData()

    nCols = len(FIELDS)
    fig, axes = plt.subplots(
        2, nCols,
        figsize=(3.0 * nCols, 6.4),
        gridspec_kw={"hspace": 0.20, "wspace": 0.05},
    )
    fig.suptitle(
        "SPH vs Stable Fluids · 50 steps",
        color=FG_PRIMARY, fontsize=14, y=0.985,
        fontweight="regular",
    )
    fig.text(
        0.5, 0.945,
        "Lagrangian particle field (top) compared with Eulerian grid solve (bottom)",
        ha="center", color=FG_SECONDARY, fontsize=9.5,
    )
    addFooter(fig)

    rowLabels = ("SPH", "Stable Fluids")
    dataSets = (sphData, sfData)

    for col, (field, label) in enumerate(zip(FIELDS, FIELD_LABELS)):
        cmap = PALETTES["diverging"] if field == "divergence" else PALETTES["sequential"]
        axes[0, col].text(
            0.5, 1.06, label,
            transform=axes[0, col].transAxes,
            ha="center", va="bottom",
            color=FG_PRIMARY, fontsize=11,
        )
        for row, data in enumerate(dataSets):
            ax = axes[row, col]
            arr = data[field]
            vAbs = float(np.percentile(np.abs(arr + 1e-9), 97))
            if field == "divergence":
                vmin, vmax = -vAbs, vAbs
            else:
                vmin, vmax = 0.0, max(vAbs, 1e-6)
            ax.imshow(
                arr, origin="lower",
                cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="bilinear",
            )
            ax.axis("off")

    for row, label in enumerate(rowLabels):
        axes[row, 0].text(
            -0.04, 0.5, label,
            transform=axes[row, 0].transAxes,
            ha="right", va="center",
            color=FG_SECONDARY, fontsize=11, rotation=90,
        )

    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.04)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {OUTPUT}")
