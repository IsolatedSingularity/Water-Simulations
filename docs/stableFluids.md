# Stable Fluids Solver

`watersim/solvers/stableFluids.py` implements the canonical Eulerian grid solver used by four scenes: Kármán vortex street, swirl, Rayleigh-Taylor instability, and lid-driven cavity.

---

## Overview

Stable Fluids (Stam 1999) is an unconditionally stable Eulerian method on a regular grid. It supports optional obstacle masks, passive scalar transport, and configurable top-wall velocity for lid-driven cavity benchmarks.

---

## Solver Parameters

| Parameter | Default | Description |
|---|---|---|
| `N` | 128 | Grid side length (cells) |
| `dt` | 0.1 | Time step |
| `diff` | 0.0 | Diffusion coefficient |
| `visc` | 0.0 | Kinematic viscosity |
| `obstacle` | None | Boolean mask array, True = solid |
| `topWallVelocity` | 0.0 | Lid velocity for cavity benchmark |

---

## Algorithm

Each call to `step()` runs:

1. **Velocity step**: diffuse, project, self-advect, project.
2. **Density step**: diffuse, advect through velocity field.
3. **Scalar step** (optional): passive scalar advected by velocity.

### Semi-Lagrangian Advection

For each grid cell $(i, j)$, trace a particle backward:

$$\mathbf{x}_0 = (i, j) - \Delta t \cdot \mathbf{u}(i, j)$$

Interpolate bilinearly at $\mathbf{x}_0$ to get the new field value. Unconditionally stable for all time step sizes.

### Pressure Projection

Compute divergence $d_{i,j} = \partial u / \partial x + \partial v / \partial y$ and solve the Poisson system $\nabla^2 p = d$ with 20 Jacobi iterations. Subtract $\nabla p$ from velocity.

### Boundary Conditions

- **No-slip walls**: velocity set to zero at all four edges.
- **Lid-driven cavity**: top wall set to `topWallVelocity` in x-direction.
- **Obstacle mask**: velocities zeroed inside obstacle cells; boundary cells enforce no-penetration.

---

## Scene Variants

| Scene | Grid | `visc` | `topWallVelocity` | Notes |
|---|---|---|---|---|
| Kármán street | 256x64 | 0.0001 | 0.0 | Cylinder obstacle, inflow source |
| Swirl | 128x128 | 0.0 | 0.0 | Two algorithmic density sources |
| Rayleigh-Taylor | 192x288 | 0.0005 | 0.0 | Buoyancy term, periodic L/R BCs |
| Lid-driven cavity | 192x192 | 0.005 | 1.0 | $Re \approx 38400$ |

---

## Diagnostics

- `getVorticity()`: returns $\omega = \partial v / \partial x - \partial u / \partial y$ as a 2D array.
- `getDivergence()`: returns $\nabla \cdot \mathbf{u}$ post-projection (should be near zero).
- `getGriddedData()`: returns a dict of `density`, `u`, `v`, `pressure`, `divergence`, `speed` arrays.

---

## See Also

- [theory.md](theory.md) for the projection method derivation
- `watersim/scenes/karmanStreet.py`, `watersim/scenes/lidDrivenCavity.py`
