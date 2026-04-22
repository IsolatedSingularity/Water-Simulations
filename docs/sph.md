# SPH Solver

`watersim/solvers/sph.py` implements a vectorized 2D SPH solver for the static comparison scene.

---

## Overview

Smoothed Particle Hydrodynamics (SPH) is a Lagrangian method: fluid properties are carried by particles that move through space. No grid is required. The method handles free-surface flows and large deformations naturally.

---

## Solver Parameters

| Parameter | Value | Description |
|---|---|---|
| `SMOOTHING_RADIUS` | 0.8 | Kernel support radius $h$ |
| `REST_DENSITY` | 300.0 | Target density $\rho_0$ |
| `GAS_CONST` | 2000.0 | Equation-of-state stiffness $k$ |
| `VISCOSITY` | 200.0 | Kinematic viscosity $\mu$ |
| `PARTICLE_MASS` | 1.0 | Per-particle mass $m$ |
| `DT` | 0.04 | Time step |
| `GRID_SIZE` | 64 | Domain side length |

---

## Algorithm

### 1. Density and Pressure

Density at particle $i$:

$$\rho_i = \sum_j m_j W_{\text{poly6}}(|\mathbf{r}_i - \mathbf{r}_j|^2, h)$$

Pressure via Tait equation of state:

$$p_i = k(\rho_i - \rho_0)$$

### 2. Pressure Force

$$\mathbf{F}_i^{\text{press}} = -\sum_j m_j \frac{p_i + p_j}{2\rho_j} \nabla W_{\text{spiky}}(\mathbf{r}_i - \mathbf{r}_j, h)$$

### 3. Viscosity Force

$$\mathbf{F}_i^{\text{visc}} = \mu \sum_j m_j \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_j} \nabla^2 W_{\text{visc}}(|\mathbf{r}_i - \mathbf{r}_j|, h)$$

### 4. Integration

Explicit Euler integration with gravity:

$$\mathbf{v}_i \leftarrow \mathbf{v}_i + \Delta t \left(\mathbf{F}_i / \rho_i + \mathbf{g}\right)$$

$$\mathbf{r}_i \leftarrow \mathbf{r}_i + \Delta t \, \mathbf{v}_i$$

---

## Vectorization

Neighbor queries use `scipy.spatial.cKDTree.query_ball_point`, which runs in $O(N \log N)$. Force accumulation is fully vectorized with NumPy array operations, avoiding explicit Python loops over particles.

---

## Grid Interpolation

`getGriddedData()` returns a 64x64 grid of density, pressure, divergence, and speed by linear binning of particle properties. Used by `staticAnalysis.py` for the side-by-side comparison plot.

---

## See Also

- [theory.md](theory.md) for kernel math
- `watersim/solvers/kernels.py` for the three kernel implementations
