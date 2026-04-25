"""
Stable Fluids solver (Stam 1999).

One canonical implementation reused by all Eulerian-based simulations:
realtime interactive, saved swirl, Karman vortex street, Rayleigh-Taylor,
and lid-driven cavity.
"""

import numpy as np

from watersim.solvers.base import Solver


class StableFluidsSolver(Solver):
    """
    2D grid-based Stable Fluids solver.

    Supports:
    - Density advection + diffusion.
    - Velocity diffusion, advection, and pressure projection.
    - Obstacle mask (boolean array) for blocked cells.
    - Optional passive scalar field for multi-fluid scenarios.
    - No-slip boundary for lid-driven cavity (topWallVelocity).
    """

    def __init__(
        self,
        size: int = 128,
        dt: float = 0.1,
        diff: float = 0.0,
        visc: float = 1e-6,
        topWallVelocity: float = 0.0,
        obstacleMask: "np.ndarray | None" = None,
    ) -> None:
        """
        Parameters
        ----------
        size:             grid resolution (size x size).
        dt:               time step.
        diff:             density diffusion coefficient.
        visc:             velocity viscosity.
        topWallVelocity:  non-zero activates lid-driven cavity mode.
        obstacleMask:     boolean (size x size) array; True = solid cell.
        """
        self.size = size
        self.dt = dt
        self.diff = diff
        self.visc = visc
        self.topWallVelocity = topWallVelocity

        self.u: np.ndarray = np.zeros((size, size), dtype=np.float64)
        self.v: np.ndarray = np.zeros((size, size), dtype=np.float64)
        self.uPrev: np.ndarray = np.zeros_like(self.u)
        self.vPrev: np.ndarray = np.zeros_like(self.v)
        self.density: np.ndarray = np.zeros((size, size), dtype=np.float64)
        self.densityPrev: np.ndarray = np.zeros_like(self.density)

        # Optional passive scalar (for Rayleigh-Taylor / multi-fluid)
        self.scalar: np.ndarray = np.zeros((size, size), dtype=np.float64)
        self.scalarPrev: np.ndarray = np.zeros_like(self.scalar)

        if obstacleMask is not None:
            self.obstacle: np.ndarray = obstacleMask.astype(bool)
        else:
            self.obstacle = np.zeros((size, size), dtype=bool)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def addSource(self, field: np.ndarray, source: np.ndarray) -> None:
        """field += dt * source (in-place)."""
        field += self.dt * source

    def _setBoundaries(self, b: int, x: np.ndarray) -> None:
        """
        Enforce boundary conditions.

        b=0: density/scalar (zero-gradient).
        b=1: reflect x-velocity at x walls.
        b=2: reflect y-velocity at y walls.
        """
        n = self.size
        x[0, :] = -x[1, :] if b == 1 else x[1, :]
        x[n - 1, :] = -x[n - 2, :] if b == 1 else x[n - 2, :]
        x[:, 0] = -x[:, 1] if b == 2 else x[:, 1]
        x[:, n - 1] = -x[:, n - 2] if b == 2 else x[:, n - 2]

        # Lid (top wall): force u = topWallVelocity for lid-driven cavity
        if b == 1 and self.topWallVelocity != 0.0:
            x[:, n - 1] = 2.0 * self.topWallVelocity - x[:, n - 2]

        # Obstacle: zero velocity inside solid cells
        if np.any(self.obstacle):
            x[self.obstacle] = 0.0

        # Corners
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, n - 1] = 0.5 * (x[1, n - 1] + x[0, n - 2])
        x[n - 1, 0] = 0.5 * (x[n - 2, 0] + x[n - 1, 1])
        x[n - 1, n - 1] = 0.5 * (x[n - 2, n - 1] + x[n - 1, n - 2])

    def _linearSolve(
        self, b: int, x: np.ndarray, x0: np.ndarray, a: float, c: float, iters: int = 20
    ) -> None:
        """Jacobi iteration for diffusion / pressure solve (in-place on x)."""
        cInv = 1.0 / c
        for _ in range(iters):
            x[1:-1, 1:-1] = (
                x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
            ) * cInv
            self._setBoundaries(b, x)

    def _diffuse(self, b: int, x: np.ndarray, x0: np.ndarray, rate: float) -> None:
        """Diffuse field x from x0."""
        n = self.size
        a = self.dt * rate * (n - 2) * (n - 2)
        self._linearSolve(b, x, x0, a, 1.0 + 4.0 * a)

    def _advect(self, b: int, d: np.ndarray, d0: np.ndarray) -> None:
        """Semi-Lagrangian advection of d0 using current (u, v) into d."""
        n = self.size
        dtScale = self.dt * (n - 2)

        i = np.arange(1, n - 1)[:, np.newaxis]
        j = np.arange(1, n - 1)[np.newaxis, :]

        x = i - dtScale * self.u[1:-1, 1:-1]
        y = j - dtScale * self.v[1:-1, 1:-1]

        x = np.clip(x, 0.5, n - 1.5)
        y = np.clip(y, 0.5, n - 1.5)

        i0 = x.astype(int)
        i1 = i0 + 1
        j0 = y.astype(int)
        j1 = j0 + 1

        s1 = x - i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1

        d[1:-1, 1:-1] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (
            t0 * d0[i1, j0] + t1 * d0[i1, j1]
        )
        self._setBoundaries(b, d)

    def _project(self) -> None:
        """Hodge decomposition: remove divergence from velocity field."""
        n = self.size
        h = 1.0 / n
        p = np.zeros((n, n), dtype=np.float64)
        div = np.zeros((n, n), dtype=np.float64)

        div[1:-1, 1:-1] = (
            -0.5 * h * (self.u[2:, 1:-1] - self.u[:-2, 1:-1] + self.v[1:-1, 2:] - self.v[1:-1, :-2])
        )
        self._setBoundaries(0, div)
        self._setBoundaries(0, p)
        self._linearSolve(0, p, div, 1.0, 4.0, iters=20)

        self.u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        self.v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        self._setBoundaries(1, self.u)
        self._setBoundaries(2, self.v)

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def _velocityStep(self) -> None:
        """Full velocity update: add sources, diffuse, project, advect, project."""
        self.addSource(self.u, self.uPrev)
        self.addSource(self.v, self.vPrev)

        self.u, self.uPrev = self.uPrev, self.u
        self.v, self.vPrev = self.vPrev, self.v

        self._diffuse(1, self.u, self.uPrev, self.visc)
        self._diffuse(2, self.v, self.vPrev, self.visc)
        self._project()

        self.u, self.uPrev = self.uPrev, self.u
        self.v, self.vPrev = self.vPrev, self.v

        self._advect(1, self.u, self.uPrev)
        self._advect(2, self.v, self.vPrev)
        self._project()

    def _densityStep(self) -> None:
        """Full density update: add sources, diffuse, advect."""
        self.addSource(self.density, self.densityPrev)
        self.density, self.densityPrev = self.densityPrev, self.density
        self._diffuse(0, self.density, self.densityPrev, self.diff)
        self.density, self.densityPrev = self.densityPrev, self.density
        self._advect(0, self.density, self.densityPrev)

    def _scalarStep(self) -> None:
        """Advect passive scalar (no diffusion by default)."""
        self._advect(0, self.scalar, self.scalarPrev)
        self.scalarPrev[:] = self.scalar

    def step(self) -> None:
        """Advance velocity, density, and scalar by one time step."""
        self._velocityStep()
        self._densityStep()
        if np.any(self.scalar != 0.0) or np.any(self.scalarPrev != 0.0):
            self._scalarStep()
        # Clear source arrays
        self.uPrev[:] = 0.0
        self.vPrev[:] = 0.0
        self.densityPrev[:] = 0.0

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------

    def getVorticity(self) -> np.ndarray:
        """Return curl field: dv/dx - du/dy, shape (size, size)."""
        vort = np.zeros_like(self.u)
        vort[1:-1, 1:-1] = (self.v[1:-1, 2:] - self.v[1:-1, :-2]) * 0.5 - (
            self.u[2:, 1:-1] - self.u[:-2, 1:-1]
        ) * 0.5
        return vort

    def getDivergence(self) -> np.ndarray:
        """Return divergence of velocity field, shape (size, size)."""
        div = np.zeros_like(self.u)
        div[1:-1, 1:-1] = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) * 0.5 + (
            self.v[1:-1, 2:] - self.v[1:-1, :-2]
        ) * 0.5
        return div

    def getGriddedData(self) -> dict[str, np.ndarray]:
        """Return dict of density, pressure proxy, divergence, speed."""
        pressure = np.zeros_like(self.density)
        self._linearSolve(0, pressure, self.getDivergence(), 1.0, 4.0, iters=5)
        speed = np.sqrt(self.u**2 + self.v**2)
        return {
            "density": self.density.copy(),
            "pressure": pressure,
            "divergence": self.getDivergence(),
            "speed": speed,
        }
