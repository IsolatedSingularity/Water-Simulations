"""
PIC/FLIP hybrid solver for dam-break simulation.

P2G (particle-to-grid) scatter uses np.add.at for vectorized accumulation.
Basin reframed to 192 x 96 (2:1 aspect) so the splash propagates visibly.
"""

import numpy as np

from watersim.solvers.base import Solver


class HybridSolver(Solver):
    """
    PIC/FLIP hybrid particle-grid solver.

    Physical domain: gridWidth x gridHeight cells.
    Particles start as a dam block on the left 30% x 70% of the domain.
    """

    FLIP_ALPHA: float = 0.95  # 1 = pure FLIP, 0 = pure PIC
    DT: float = 0.04
    GRAVITY: np.ndarray = np.array([0.0, -9.8 * 2.0])

    def __init__(
        self,
        gridWidth: int = 192,
        gridHeight: int = 96,
        nParticlesPerRow: int = 50,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        gridWidth:        horizontal grid resolution.
        gridHeight:       vertical grid resolution.
        nParticlesPerRow: particles along one axis of the initial dam block.
        seed:             RNG seed for reproducibility.
        """
        np.random.seed(seed)
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

        # Initial dam block: left 30% width x 70% height
        colCount = nParticlesPerRow
        rowCount = int(nParticlesPerRow * 0.7)
        xs = np.linspace(2.0, gridWidth * 0.30, colCount)
        ys = np.linspace(2.0, gridHeight * 0.70, rowCount)
        xGrid, yGrid = np.meshgrid(xs, ys)
        self.pos: np.ndarray = np.column_stack([xGrid.ravel(), yGrid.ravel()]).astype(np.float64)
        self.vel: np.ndarray = np.zeros((self.pos.shape[0], 2), dtype=np.float64)
        self.nParticles: int = self.pos.shape[0]

        # Staggered grid: u on (gridHeight+1) x (gridWidth+1) etc.
        gH, gW = gridHeight, gridWidth
        self.gridU: np.ndarray = np.zeros((gH + 1, gW + 1), dtype=np.float64)
        self.gridV: np.ndarray = np.zeros((gH + 1, gW + 1), dtype=np.float64)

    # ------------------------------------------------------------------
    # P2G: particle -> grid  (vectorized with np.add.at)
    # ------------------------------------------------------------------

    def _particlesToGrid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Transfer particle velocities to staggered grid using bilinear weights.

        Returns old grid u, v before this transfer (used in G2P FLIP delta).
        """
        gH, gW = self.gridHeight, self.gridWidth
        uNum = np.zeros((gH + 1, gW + 1), dtype=np.float64)
        vNum = np.zeros_like(uNum)
        uDen = np.zeros_like(uNum)
        vDen = np.zeros_like(uNum)

        px = self.pos[:, 0]
        py = self.pos[:, 1]

        # U-grid: offset (0.0, 0.5) in cell-space
        ux = px
        uy = py - 0.5

        i0u = np.clip(ux.astype(int), 0, gW - 1)
        j0u = np.clip(uy.astype(int), 0, gH - 1)
        i1u = i0u + 1
        j1u = j0u + 1
        s1u = ux - i0u
        s0u = 1.0 - s1u
        t1u = uy - j0u
        t0u = 1.0 - t1u

        pu = self.vel[:, 0]
        np.add.at(uNum, (j0u, i0u), s0u * t0u * pu)
        np.add.at(uNum, (j1u, i0u), s0u * t1u * pu)
        np.add.at(uNum, (j0u, i1u), s1u * t0u * pu)
        np.add.at(uNum, (j1u, i1u), s1u * t1u * pu)
        np.add.at(uDen, (j0u, i0u), s0u * t0u)
        np.add.at(uDen, (j1u, i0u), s0u * t1u)
        np.add.at(uDen, (j0u, i1u), s1u * t0u)
        np.add.at(uDen, (j1u, i1u), s1u * t1u)

        # V-grid: offset (0.5, 0.0)
        vx = px - 0.5
        vy = py

        i0v = np.clip(vx.astype(int), 0, gW - 1)
        j0v = np.clip(vy.astype(int), 0, gH - 1)
        i1v = i0v + 1
        j1v = j0v + 1
        s1v = vx - i0v
        s0v = 1.0 - s1v
        t1v = vy - j0v
        t0v = 1.0 - t1v

        pv = self.vel[:, 1]
        np.add.at(vNum, (j0v, i0v), s0v * t0v * pv)
        np.add.at(vNum, (j1v, i0v), s0v * t1v * pv)
        np.add.at(vNum, (j0v, i1v), s1v * t0v * pv)
        np.add.at(vNum, (j1v, i1v), s1v * t1v * pv)
        np.add.at(vDen, (j0v, i0v), s0v * t0v)
        np.add.at(vDen, (j1v, i0v), s0v * t1v)
        np.add.at(vDen, (j0v, i1v), s1v * t0v)
        np.add.at(vDen, (j1v, i1v), s1v * t1v)

        validU = uDen > 0
        validV = vDen > 0
        self.gridU = np.where(validU, uNum / np.maximum(uDen, 1e-9), self.gridU)
        self.gridV = np.where(validV, vNum / np.maximum(vDen, 1e-9), self.gridV)

        return self.gridU.copy(), self.gridV.copy()

    # ------------------------------------------------------------------
    # Pressure projection (Poisson solve via Jacobi)
    # ------------------------------------------------------------------

    def _project(self) -> None:
        """Enforce incompressibility with 40-iteration Jacobi pressure solve."""
        gH, gW = self.gridHeight, self.gridWidth
        p = np.zeros((gH + 1, gW + 1), dtype=np.float64)
        div = np.zeros_like(p)

        div[1:-1, 1:-1] = (
            self.gridU[1:-1, 2:]
            - self.gridU[1:-1, 1:-1]
            + self.gridV[2:, 1:-1]
            - self.gridV[1:-1, 1:-1]
        )

        for _ in range(80):
            p[1:-1, 1:-1] = (
                div[1:-1, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1]
            ) / 4.0
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]

        self.gridU[1:-1, 1:-1] -= p[1:-1, 1:-1] - p[1:-1, :-2]
        self.gridV[1:-1, 1:-1] -= p[1:-1, 1:-1] - p[:-2, 1:-1]

        # Wall boundaries
        self.gridU[:, 0] = 0.0
        self.gridU[:, -1] = 0.0
        self.gridV[0, :] = 0.0
        self.gridV[-1, :] = 0.0

    # ------------------------------------------------------------------
    # G2P: grid -> particle  (FLIP/PIC blend)
    # ------------------------------------------------------------------

    def _gridToParticles(self, oldU: np.ndarray, oldV: np.ndarray) -> None:
        """
        Transfer grid velocities back to particles (FLIP/PIC blend).

        FLIP_ALPHA controls PIC vs FLIP split.
        """
        gH, gW = self.gridHeight, self.gridWidth
        px = np.clip(self.pos[:, 0], 0.5, gW - 0.5)
        py = np.clip(self.pos[:, 1], 0.5, gH - 0.5)

        def interpGrid(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            i0 = np.clip(x.astype(int), 0, grid.shape[1] - 2)
            j0 = np.clip(y.astype(int), 0, grid.shape[0] - 2)
            i1 = i0 + 1
            j1 = j0 + 1
            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1
            return (
                s0 * t0 * grid[j0, i0]
                + s0 * t1 * grid[j1, i0]
                + s1 * t0 * grid[j0, i1]
                + s1 * t1 * grid[j1, i1]
            )

        picU = interpGrid(self.gridU, px, py)
        picV = interpGrid(self.gridV, px, py)
        oldInterpU = interpGrid(oldU, px, py)
        oldInterpV = interpGrid(oldV, px, py)

        flipU = self.vel[:, 0] + (picU - oldInterpU)
        flipV = self.vel[:, 1] + (picV - oldInterpV)

        a = self.FLIP_ALPHA
        self.vel[:, 0] = a * flipU + (1.0 - a) * picU
        self.vel[:, 1] = a * flipV + (1.0 - a) * picV

    # ------------------------------------------------------------------
    # Boundary enforcement for particles
    # ------------------------------------------------------------------

    def _applyBoundaries(self) -> None:
        """Elastic collision response at domain walls (damping = -0.5)."""
        gW, gH = self.gridWidth, self.gridHeight

        for dim, limit in [(0, gW), (1, gH)]:
            overMax = self.pos[:, dim] > limit - 1.0
            self.vel[overMax, dim] *= -0.5
            self.pos[overMax, dim] = limit - 1.0

            underMin = self.pos[:, dim] < 1.0
            self.vel[underMin, dim] *= -0.5
            self.pos[underMin, dim] = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """One PIC/FLIP step: P2G, gravity, project, G2P, advect, boundaries."""
        oldU, oldV = self._particlesToGrid()
        # Apply gravity to grid velocity
        self.gridV += self.DT * self.GRAVITY[1]
        self._project()
        self._gridToParticles(oldU, oldV)
        # Clamp velocities to prevent runaway instability
        self.vel = np.clip(self.vel, -60.0, 60.0)
        # Advect particles
        self.pos += self.DT * self.vel
        self._applyBoundaries()

    def getParticlePositions(self) -> np.ndarray:
        """Return particle positions, shape (N, 2)."""
        return self.pos.copy()

    def getParticleSpeeds(self) -> np.ndarray:
        """Return per-particle speed magnitudes, shape (N,)."""
        return np.linalg.norm(self.vel, axis=1)
