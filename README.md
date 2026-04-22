# 2D Fluid Dynamics Lab

![Kármán Vortex Street](Plots/vortex_street.gif)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Compact 2D incompressible fluid simulation framework comparing three methods: Lagrangian SPH, Eulerian Stable Fluids, and Hybrid PIC/FLIP. All simulations run in pure NumPy + SciPy on a standard laptop.

---

## Table of Contents

1. [Methods](#methods)
2. [Gallery](#gallery)
3. [Installation](#installation)
4. [Running Simulations](#running-simulations)
5. [Project Structure](#project-structure)
6. [Theory and References](#theory-and-references)

---

## Methods

### Stable Fluids (Eulerian)

Grid-based solver. Unconditionally stable via semi-Lagrangian advection and pressure projection.

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{\nabla p}{\rho} + \nu \nabla^2 \mathbf{u} + \mathbf{f}, \quad \nabla \cdot \mathbf{u} = 0$$

Used by: Kármán vortex street, swirl animation, Rayleigh-Taylor instability, lid-driven cavity, realtime interactive.

### SPH (Lagrangian)

Particle-based solver. Fluid properties estimated via weighted kernel sums over neighbors within radius $h$:

$$\rho_i = \sum_j m_j W_{\text{poly6}}(|\mathbf{r}_i - \mathbf{r}_j|^2, h)$$

Neighbor queries via `scipy.spatial.cKDTree` for $O(N \log N)$ scaling. Used by: static comparison.

### PIC/FLIP (Hybrid)

Particles carry velocity (no advective diffusion); grid enforces incompressibility. FLIP velocity update:

$$\mathbf{v}_p^{n+1} = \alpha\left(\mathbf{v}_p^n + \Delta\mathbf{v}_{\text{grid}}\right) + (1-\alpha)\,\mathbf{v}_{\text{grid}}^{n+1}$$

$\alpha = 0.95$. P2G transfer uses `np.add.at` for vectorized scatter-add. Used by: dam-break animation.

---

## Gallery

| Scene | Preview | Method |
|---|---|---|
| Kármán Vortex Street | ![vortex](Plots/vortex_street.gif) | Stable Fluids, cylinder obstacle |
| Two-Source Swirl | ![swirl](Plots/saved_simulation.gif) | Stable Fluids, algorithmic sources |
| PIC/FLIP Dam Break | ![dam](Plots/hybrid_simulation.gif) | PIC/FLIP hybrid, 192x96 basin |
| Rayleigh-Taylor Instability | ![rt](Plots/rayleigh_taylor.gif) | Stable Fluids, buoyancy + passive scalar |
| Lid-Driven Cavity | ![lid](Plots/lid_driven_cavity.gif) | Stable Fluids, $Re \approx 38400$ |
| Static Comparison | ![static](Plots/static_comparison.png) | SPH vs Stable Fluids snapshot |

---

## Installation

```bash
git clone https://github.com/IsolatedSingularity/Water-Simulations
cd Water-Simulations
pip install -r requirements.txt
```

Python 3.11 or later required. No GPU, no compiled extensions.

---

## Running Simulations

All scripts are thin entry points under `scripts/`. Run from the repo root:

```bash
python -m scripts.runKarmanStreet      # Kármán vortex street (500 frames)
python -m scripts.runSavedSwirl        # Two-source swirl (360 frames)
python -m scripts.runPicFlipDamBreak   # PIC/FLIP dam break (500 frames)
python -m scripts.runRayleighTaylor    # Rayleigh-Taylor instability (600 frames)
python -m scripts.runLidDrivenCavity   # Lid-driven cavity (600 frames)
python -m scripts.runStaticAnalysis    # SPH vs Stable Fluids comparison (PNG)
python -m scripts.runRealtime          # Interactive Stable Fluids (mouse input)
```

Output GIFs and PNG are saved to `Plots/`. Running time is roughly 1-5 minutes per script on a standard laptop.

### Tests

```bash
python -m pytest tests/ -v
```

18 tests covering kernel correctness and solver smoke tests.

---

## Project Structure

```
watersim/               Package
  solvers/
    kernels.py          SPH kernel functions (poly6, spiky, viscosity)
    sph.py              SPHSolver
    stableFluids.py     StableFluidsSolver
    picFlip.py          HybridSolver (PIC/FLIP)
    base.py             Abstract Solver base class
  scenes/
    karmanStreet.py     Kármán vortex street
    swirl.py            Two-source swirl
    damBreak.py         PIC/FLIP dam break
    rayleighTaylor.py   Rayleigh-Taylor instability
    lidDrivenCavity.py  Lid-driven cavity
    staticAnalysis.py   SPH vs Stable Fluids comparison
    realtime.py         Interactive simulation
  viz/
    theme.py            Dark theme, palettes, footer
    overlays.py         Streamline and vorticity overlays
    animator.py         Shared FuncAnimation wrapper
  config.py             Dataclass-based scene configs
scripts/                Thin entry points (~5 lines each)
tests/                  pytest test suite
docs/                   Per-solver deep-dive documentation
  theory.md             NS equations, projection, kernels, PIC/FLIP
  sph.md                SPH solver details
  stableFluids.md       Stable Fluids solver details
  picFlip.md            PIC/FLIP solver details
```

---

## Theory and References

Full derivations in [docs/theory.md](docs/theory.md). Per-solver details in [docs/sph.md](docs/sph.md), [docs/stableFluids.md](docs/stableFluids.md), [docs/picFlip.md](docs/picFlip.md).

- Stam, J. (1999). *Stable Fluids*. SIGGRAPH 99.
- Müller, M., Charypar, D., & Gross, M. (2003). *Particle-Based Fluid Simulation for Interactive Applications*. SCA 2003.
- Zhu, Y. & Bridson, R. (2005). *Animating Sand as a Fluid*. SIGGRAPH 2005.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). *High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method*. J. Comput. Phys. 48.
- Sharp, D.H. (1984). *An overview of Rayleigh-Taylor instability*. Physica D: Nonlinear Phenomena 12.