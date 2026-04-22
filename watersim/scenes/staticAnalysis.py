"""
Static comparison scene: SPH vs Stable Fluids side-by-side snapshot.

4 columns per solver: density, pressure, divergence, speed.
Output: Plots/static_comparison.png at 300 DPI.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from watersim.solvers.sph import SPHSolver
from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.viz.theme import applyDarkTheme, addFooter, PALETTES
from watersim.viz.animator import ensurePlotDir, PLOT_DIR


SIMULATION_STEPS = 50
N_PER_ROW = 30   # ~1800 particles; feasible with cKDTree vectorization
OUTPUT = os.path.join(PLOT_DIR, "static_comparison.png")
FIELDS = ("density", "pressure", "divergence", "speed")
FIELD_LABELS = ("Density", "Pressure", "Divergence", "Speed")


def runStaticAnalysis() -> None:
    """Run both solvers for SIMULATION_STEPS and save a 2x4 comparison PNG."""
    applyDarkTheme()
    ensurePlotDir()

    # --- SPH ---
    sphSolver = SPHSolver(nParticlesPerRow=N_PER_ROW)
    for _ in range(SIMULATION_STEPS):
        sphSolver.step()
    sphData = sphSolver.getGriddedData()

    # --- Stable Fluids ---
    sfSolver = StableFluidsSolver(size=64)
    # Seed with dam-break density
    sfSolver.density[:, : sfSolver.size // 3] = 1.0
    for _ in range(SIMULATION_STEPS):
        sfSolver.step()
    sfData = sfSolver.getGriddedData()

    # --- Plot ---
    nCols = len(FIELDS)
    fig, axes = plt.subplots(2, nCols, figsize=(4 * nCols, 8))
    fig.suptitle(
        "SPH (top)  vs  Stable Fluids (bottom)  ·  50 steps",
        color="#e6edf3",
        fontsize=14,
        fontweight="semibold",
        y=0.98,
    )
    addFooter(fig)

    rowLabels = ["SPH", "Stable Fluids"]
    dataSets = [sphData, sfData]

    for row, (data, rowLabel) in enumerate(zip(dataSets, rowLabels)):
        for col, (field, label) in enumerate(zip(FIELDS, FIELD_LABELS)):
            ax = axes[row, col]
            arr = data[field]
            cmap = PALETTES["diverging"] if field == "divergence" else PALETTES["sequential"]
            vAbs = float(np.percentile(np.abs(arr + 1e-9), 99))
            if field == "divergence":
                im = ax.imshow(arr, origin="lower", cmap=cmap,
                               vmin=-vAbs, vmax=vAbs, interpolation="bilinear")
            else:
                im = ax.imshow(arr, origin="lower", cmap=cmap,
                               vmin=0, vmax=max(vAbs, 1e-6), interpolation="bilinear")
            ax.set_title(
                f"{rowLabel}: {label}" if col == 0 else label,
                color="#e6edf3", fontsize=11
            )
            ax.axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")
