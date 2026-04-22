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

All three solvers model a 2D incompressible Newtonian fluid. The governing equations are the incompressible Navier-Stokes system:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{\nabla p}{\rho} + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

$$\nabla \cdot \mathbf{u} = 0$$

The first equation is Newton's second law for a fluid parcel: the left side captures inertia and convective acceleration, the right side captures pressure gradients, viscous diffusion, and external body forces. The second equation is incompressibility: the fluid neither compresses nor expands, so the velocity field must be divergence-free at every point.

### Pressure Projection

The core challenge of incompressible simulation is enforcing $\nabla \cdot \mathbf{u} = 0$ at every time step. The standard approach uses the Helmholtz decomposition theorem: any vector field $\mathbf{w}$ can be uniquely split into a divergence-free part and a gradient part,

$$\mathbf{w} = \mathbf{u} + \nabla p$$

Taking the divergence of both sides and applying $\nabla \cdot \mathbf{u} = 0$ gives the Poisson equation $\nabla^2 p = \nabla \cdot \mathbf{w}$. Solving for $p$ and subtracting $\nabla p$ from $\mathbf{w}$ yields the pressure-corrected, divergence-free velocity. This is the projection step used by both `StableFluidsSolver` and `HybridSolver`.

### Stable Fluids (Eulerian)

Stam's key insight was to use a semi-Lagrangian advection scheme that is unconditionally stable for any time step. Rather than pushing fluid forward, we trace each grid cell backward along the velocity field:

$$\phi^{n+1}(\mathbf{x}) = \phi^n(\mathbf{x} - \mathbf{u}\,\Delta t)$$

The value at position $\mathbf{x}$ at the next time step equals the value at the back-traced position at the current time, found by bilinear interpolation. This eliminates the CFL stability constraint and allows real-time frame rates. The tradeoff is that the scheme is first-order accurate and numerically dissipative, which is why vortical structures gradually decay over long runs.

### SPH (Lagrangian)

SPH replaces the continuous fluid with a set of particles, each carrying mass, position, velocity, and pressure. Any field quantity $A$ can be estimated at any point by a weighted sum over nearby particles:

$$A(\mathbf{r}) = \sum_j \frac{m_j}{\rho_j} A_j \, W(|\mathbf{r} - \mathbf{r}_j|, h)$$

where $W$ is a smooth kernel with compact support radius $h$. The Poly6 kernel is used for density:

$$W_{\text{poly6}}(r^2, h) = \frac{315}{64\pi h^9}(h^2 - r^2)^3$$

Pressure forces use the gradient of the Spiky kernel (chosen for its non-zero gradient near $r = 0$, which prevents particle clumping):

$$\nabla W_{\text{spiky}}(\mathbf{r}, h) = -\frac{45}{\pi h^6}(h - r)^2 \frac{\mathbf{r}}{r}$$

Each particle's pressure is computed from a Tait equation of state $p_i = k(\rho_i - \rho_0)$, which drives particles apart when density exceeds the rest density $\rho_0$. Neighbor queries use `scipy.spatial.cKDTree` for $O(N \log N)$ lookup instead of $O(N^2)$ brute force.

### PIC/FLIP (Hybrid)

Pure Lagrangian methods handle free surfaces naturally but struggle with incompressibility. Pure Eulerian methods enforce incompressibility cheaply but suffer advective diffusion. PIC/FLIP (Particle-in-Cell / Fluid-Implicit-Particle) takes the best of both.

Particles carry velocity throughout the simulation, so there is no numerical diffusion from advection. At each step, particle velocities are scattered to a background MAC grid (P2G), the grid enforces incompressibility via pressure projection, then the corrected velocities are read back to the particles (G2P). The FLIP update sends only the *change* in grid velocity back to the particles, preserving small-scale kinetic energy:

$$\mathbf{v}_p^{n+1} = \alpha\underbrace{\left(\mathbf{v}_p^n + \Delta\mathbf{v}_{\text{grid}}\right)}_{\text{FLIP}} + (1-\alpha)\underbrace{\mathbf{v}_{\text{grid}}^{n+1}}_{\text{PIC}}$$

With $\alpha = 0.95$, the solver is 95% FLIP (energetic, detail-preserving) blended with 5% PIC (damping, stabilizing). The P2G scatter uses `np.add.at` for vectorized bilinear accumulation on a staggered MAC grid.

### Phenomena

**Kármán vortex street:** When a viscous flow passes a blunt obstacle, boundary layer separation causes the wake to roll up into alternating vortices. The shedding frequency is governed by the Strouhal number $St = fD/U$, where $f$ is frequency, $D$ is cylinder diameter, and $U$ is free-stream velocity. Shedding begins around $Re = 40$ and becomes periodic around $Re = 100$.

**Rayleigh-Taylor instability:** When a denser fluid sits above a lighter fluid in a gravitational field, the interface is unstable to infinitesimal perturbations. Small sinusoidal disturbances grow exponentially, forming the characteristic mushroom-cap spikes and bubbles. The passive scalar field tracks which fluid parcel originated in the dense upper layer.

**Lid-driven cavity:** A square cavity with a moving top wall and three no-slip walls is a canonical benchmark for incompressible solvers. The moving lid drives a large primary vortex that fills the cavity, with secondary Moffatt eddies forming in the bottom corners. The Ghia et al. 1982 paper gives reference velocity profiles at several Reynolds numbers.

### References

- Stam, J. (1999). *Stable Fluids*. SIGGRAPH 99.
- Müller, M., Charypar, D., & Gross, M. (2003). *Particle-Based Fluid Simulation for Interactive Applications*. SCA 2003.
- Zhu, Y. & Bridson, R. (2005). *Animating Sand as a Fluid*. SIGGRAPH 2005.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). *High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method*. J. Comput. Phys. 48.
- Sharp, D.H. (1984). *An overview of Rayleigh-Taylor instability*. Physica D: Nonlinear Phenomena 12.