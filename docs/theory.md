# Theory: 2D Incompressible Fluid Simulation

This document covers the shared mathematical foundations used across all three solvers in the `watersim` package.

---

## Navier-Stokes Equations

All simulations model an incompressible Newtonian fluid governed by the Navier-Stokes equations:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{\nabla p}{\rho} + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

$$\nabla \cdot \mathbf{u} = 0$$

where $\mathbf{u}$ is the velocity field, $p$ is pressure, $\rho$ is density, $\nu$ is kinematic viscosity, and $\mathbf{f}$ represents body forces (gravity).

---

## Projection Method (Stable Fluids, PIC/FLIP)

The incompressibility constraint $\nabla \cdot \mathbf{u} = 0$ is enforced via Helmholtz decomposition. Any vector field $\mathbf{w}$ can be split as:

$$\mathbf{w} = \mathbf{u} + \nabla p$$

where $\mathbf{u}$ is divergence-free. Taking the divergence: $\nabla^2 p = \nabla \cdot \mathbf{w}$ (Poisson equation). The corrected velocity is $\mathbf{u} = \mathbf{w} - \nabla p$.

In the `StableFluidsSolver`, this Poisson equation is solved with a Jacobi iterative scheme (20 iterations). The `HybridSolver` uses 80 iterations over a 193x97 staggered grid.

**Semi-Lagrangian advection** (unconditionally stable, Stam 1999):

$$\phi^{n+1}(\mathbf{x}) = \phi^n(\mathbf{x} - \mathbf{u} \Delta t)$$

The fluid property $\phi$ at position $\mathbf{x}$ is traced back one time step and interpolated bilinearly.

---

## SPH Kernels

SPH (Smoothed Particle Hydrodynamics) estimates field quantities as weighted sums over neighboring particles within smoothing radius $h$:

$$A_i = \sum_j \frac{m_j}{\rho_j} A_j W(|\mathbf{r}_i - \mathbf{r}_j|, h)$$

Three kernels are implemented in `watersim/solvers/kernels.py`:

### Poly6 (density estimation)

$$W_{\text{poly6}}(r^2, h) = \frac{315}{64\pi h^9}(h^2 - r^2)^3, \quad r \leq h$$

### Spiky gradient (pressure forces)

$$\nabla W_{\text{spiky}}(\mathbf{r}, h) = -\frac{45}{\pi h^6}(h - r)^2 \frac{\mathbf{r}}{r}, \quad r \leq h$$

### Viscosity Laplacian (viscous forces)

$$\nabla^2 W_{\text{visc}}(r, h) = \frac{45}{\pi h^6}(h - r), \quad r \leq h$$

---

## PIC/FLIP Hybrid

PIC/FLIP blends two velocity update strategies:

$$\mathbf{v}_p^{n+1} = \alpha \left(\mathbf{v}_p^n + \Delta \mathbf{v}_{\text{grid}}\right) + (1-\alpha)\, \mathbf{v}_{\text{grid}}^{n+1}$$

where $\alpha \in [0, 1]$ is `FLIP_ALPHA`. $\alpha = 1$ is pure FLIP (no numerical dissipation, energetic), $\alpha = 0$ is pure PIC (overdamped). The package uses $\alpha = 0.95$.

Particle-to-grid (P2G) transfer uses bilinear weights on a staggered MAC grid. Grid-to-particle (G2P) uses the same weights in reverse.

---

## Reynolds Number

For the Kármán vortex street and lid-driven cavity:

$$Re = \frac{U L}{\nu}$$

where $U$ is the characteristic velocity, $L$ is the characteristic length (cylinder diameter or cavity side), and $\nu$ is viscosity. Kármán shedding begins at $Re \approx 40$. The lid-driven cavity develops a steady primary vortex at $Re \approx 100$ and secondary corner vortices at higher $Re$.

---

## References

- Stam, J. (1999). *Stable Fluids*. SIGGRAPH 99.
- Müller, M., Charypar, D., & Gross, M. (2003). *Particle-Based Fluid Simulation for Interactive Applications*. SCA 2003.
- Zhu, Y. & Bridson, R. (2005). *Animating Sand as a Fluid*. SIGGRAPH 2005.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). *High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method*. J. Comput. Phys.
- Sharp, D.H. (1984). *An overview of Rayleigh-Taylor instability*. Physica D.
