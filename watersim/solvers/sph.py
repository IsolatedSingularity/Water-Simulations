"""SPH (Smoothed Particle Hydrodynamics) solver, vectorized with cKDTree."""

import numpy as np
from scipy.spatial import cKDTree

from watersim.solvers.base import Solver
from watersim.solvers.kernels import poly6Kernel, spikyGradKernel, viscLaplacianKernel


class SPHSolver(Solver):
    """
    2D SPH dam-break solver.

    Uses cKDTree for O(N log N) neighbor queries. camelCase throughout.
    """

    # Physical constants
    GRAVITY: np.ndarray = np.array([0.0, -9.8])
    GAS_CONST: float = 2000.0
    REST_DENSITY: float = 300.0
    VISCOSITY: float = 200.0
    SMOOTHING_RADIUS: float = 0.8
    PARTICLE_MASS: float = 1.0
    DT: float = 0.04
    GRID_SIZE: int = 64

    def __init__(self, nParticlesPerRow: int = 20, seed: int = 42) -> None:
        """
        Initialise particles in a dam-break block on the left side.

        Parameters
        ----------
        nParticlesPerRow: particles along each axis of the initial block.
        seed:             random seed for reproducibility (unused here but kept
                          for API consistency with scene callers).
        """
        np.random.seed(seed)
        nParticles = nParticlesPerRow * nParticlesPerRow
        xs = np.linspace(0.1, 0.4, nParticlesPerRow)
        ys = np.linspace(0.1, 0.9, nParticlesPerRow)
        xGrid, yGrid = np.meshgrid(xs, ys)
        self.pos: np.ndarray = np.column_stack(
            [xGrid.ravel(), yGrid.ravel()]
        ).astype(np.float64)
        self.vel: np.ndarray = np.zeros((nParticles, 2), dtype=np.float64)
        self.acc: np.ndarray = np.zeros((nParticles, 2), dtype=np.float64)
        self.rho: np.ndarray = np.zeros(nParticles, dtype=np.float64)
        self.pressure: np.ndarray = np.zeros(nParticles, dtype=np.float64)
        self.nParticles: int = nParticles
        self.h: float = self.SMOOTHING_RADIUS

    # ------------------------------------------------------------------
    # Internal update methods
    # ------------------------------------------------------------------

    def _updateFluidProperties(self) -> None:
        """Compute density and pressure for every particle via cKDTree query."""
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(self.h, output_type="ndarray")

        self.rho[:] = 0.0
        # self-contribution
        self.rho += self.PARTICLE_MASS * poly6Kernel(
            np.zeros(self.nParticles), self.h
        )

        if pairs.shape[0] > 0:
            i, j = pairs[:, 0], pairs[:, 1]
            diff = self.pos[i] - self.pos[j]
            rSq = np.sum(diff ** 2, axis=1)
            w = self.PARTICLE_MASS * poly6Kernel(rSq, self.h)
            np.add.at(self.rho, i, w)
            np.add.at(self.rho, j, w)

        self.pressure[:] = self.GAS_CONST * (self.rho - self.REST_DENSITY)

    def _computeForces(self) -> None:
        """
        Compute pressure + viscosity forces.

        Division by rho is guarded: skipped when rho <= 1e-9.
        """
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(self.h, output_type="ndarray")

        forces = np.zeros((self.nParticles, 2), dtype=np.float64)

        if pairs.shape[0] > 0:
            i, j = pairs[:, 0], pairs[:, 1]
            diff = self.pos[i] - self.pos[j]
            r = np.linalg.norm(diff, axis=1)

            # Pressure force
            validRho = (self.rho[i] > 1e-9) & (self.rho[j] > 1e-9)
            if np.any(validRho):
                avgPressure = (
                    self.pressure[i[validRho]] / self.rho[i[validRho]]
                    + self.pressure[j[validRho]] / self.rho[j[validRho]]
                ) * 0.5
                gradW = spikyGradKernel(diff[validRho], r[validRho], self.h)
                pressForce = (
                    -self.PARTICLE_MASS * avgPressure[:, np.newaxis] * gradW
                )
                np.add.at(forces, i[validRho], pressForce)
                np.add.at(forces, j[validRho], -pressForce)

            # Viscosity force
            validVisc = (self.rho[j] > 1e-9)
            if np.any(validVisc):
                lapW = viscLaplacianKernel(r[validVisc], self.h)
                velDiff = self.vel[j[validVisc]] - self.vel[i[validVisc]]
                viscForce = (
                    self.VISCOSITY
                    * self.PARTICLE_MASS
                    * velDiff
                    * (lapW / self.rho[j[validVisc]])[:, np.newaxis]
                )
                np.add.at(forces, i[validVisc], viscForce)
                np.add.at(forces, j[validVisc], -viscForce)

        # Gravity
        self.acc = forces / np.maximum(self.rho[:, np.newaxis], 1e-9)
        self.acc += self.GRAVITY

    def _integrate(self) -> None:
        """Semi-implicit Euler integration + hard wall boundaries [0, 1]."""
        self.vel += self.DT * self.acc
        self.pos += self.DT * self.vel
        # Reflect + clip at walls
        for dim in range(2):
            overMax = self.pos[:, dim] > 1.0
            self.vel[overMax, dim] *= -0.5
            self.pos[overMax, dim] = 1.0
            underMin = self.pos[:, dim] < 0.0
            self.vel[underMin, dim] *= -0.5
            self.pos[underMin, dim] = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance by one SPH time step."""
        self._updateFluidProperties()
        self._computeForces()
        self._integrate()

    def getGriddedData(self) -> dict[str, np.ndarray]:
        """
        Interpolate particle properties onto a uniform grid.

        Returns
        -------
        Dict with keys 'density', 'pressure', 'divergence', 'speed'.
        """
        gs = self.GRID_SIZE
        grid = np.linspace(0, 1, gs)

        fields: dict[str, np.ndarray] = {
            k: np.zeros((gs, gs)) for k in ("density", "pressure", "divergence", "speed")
        }
        counts = np.zeros((gs, gs))

        ix = np.clip((self.pos[:, 0] * (gs - 1)).astype(int), 0, gs - 1)
        iy = np.clip((self.pos[:, 1] * (gs - 1)).astype(int), 0, gs - 1)

        np.add.at(fields["density"], (iy, ix), self.rho)
        np.add.at(fields["pressure"], (iy, ix), self.pressure)
        np.add.at(fields["speed"], (iy, ix),
                  np.linalg.norm(self.vel, axis=1))
        np.add.at(counts, (iy, ix), 1)

        mask = counts > 0
        for k in ("density", "pressure", "speed"):
            fields[k][mask] /= counts[mask]

        # Divergence via finite differences of interpolated velocity
        uGrid = np.zeros((gs, gs))
        vGrid = np.zeros((gs, gs))
        np.add.at(uGrid, (iy, ix), self.vel[:, 0])
        np.add.at(vGrid, (iy, ix), self.vel[:, 1])
        uGrid[mask] /= counts[mask]
        vGrid[mask] /= counts[mask]
        fields["divergence"][1:-1, 1:-1] = (
            uGrid[1:-1, 2:] - uGrid[1:-1, :-2]
            + vGrid[2:, 1:-1] - vGrid[:-2, 1:-1]
        ) * 0.5

        return fields
