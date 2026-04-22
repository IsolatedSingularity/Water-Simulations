"""Smoke tests for solver correctness: SPH mass conservation, SF step health."""

import numpy as np
import pytest

from watersim.solvers.sph import SPHSolver
from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.solvers.picFlip import HybridSolver


# ---------------------------------------------------------------------------
# SPH
# ---------------------------------------------------------------------------

def testSPHParticleCountPreserved() -> None:
    """SPH does not lose or gain particles over 5 steps."""
    solver = SPHSolver(nParticlesPerRow=5)
    nBefore = solver.nParticles
    for _ in range(5):
        solver.step()
    assert solver.pos.shape[0] == nBefore
    assert solver.vel.shape[0] == nBefore


def testSPHBoundaryEnforcement() -> None:
    """SPH particles never leave the [0, 1] domain."""
    solver = SPHSolver(nParticlesPerRow=10)
    for _ in range(20):
        solver.step()
    assert np.all(solver.pos >= 0.0), "Particles below domain lower bound."
    assert np.all(solver.pos <= 1.0), "Particles above domain upper bound."


def testSPHDensityPositive() -> None:
    """SPH density is non-negative after update."""
    solver = SPHSolver(nParticlesPerRow=5)
    solver._updateFluidProperties()
    assert np.all(solver.rho >= 0.0), "Negative density found."


def testSPHGriddedDataKeys() -> None:
    """getGriddedData returns all expected field keys."""
    solver = SPHSolver(nParticlesPerRow=5)
    solver.step()
    data = solver.getGriddedData()
    for key in ("density", "pressure", "divergence", "speed"):
        assert key in data, f"Missing key: {key}"
        assert data[key].shape == (SPHSolver.GRID_SIZE, SPHSolver.GRID_SIZE)


# ---------------------------------------------------------------------------
# Stable Fluids
# ---------------------------------------------------------------------------

def testStableFluidsStepRunsWithoutError() -> None:
    """StableFluidsSolver.step() completes 5 steps without raising."""
    solver = StableFluidsSolver(size=32)
    for _ in range(5):
        solver.step()


def testStableFluidsDensityNonNegative() -> None:
    """Density stays non-negative after seeding and 10 steps."""
    solver = StableFluidsSolver(size=32)
    solver.density[:, :16] = 1.0
    for _ in range(10):
        solver.step()
    assert np.all(solver.density >= -1e-9), "Negative density after advection."


def testStableFluidsDivergenceNearZero() -> None:
    """Divergence after projection is small (incompressibility check)."""
    solver = StableFluidsSolver(size=32)
    solver.u[8:24, 8:24] = 1.0
    solver.step()
    div = solver.getDivergence()
    assert float(np.max(np.abs(div[1:-1, 1:-1]))) < 0.5, \
        "Divergence too large after projection."


def testStableFluidsVorticityShape() -> None:
    """getVorticity returns correct shape."""
    solver = StableFluidsSolver(size=64)
    vort = solver.getVorticity()
    assert vort.shape == (64, 64)


# ---------------------------------------------------------------------------
# HybridSolver (PIC/FLIP)
# ---------------------------------------------------------------------------

def testHybridParticleCountPreserved() -> None:
    """PIC/FLIP does not change particle count over 5 steps."""
    solver = HybridSolver(gridWidth=48, gridHeight=24, nParticlesPerRow=10)
    nBefore = solver.nParticles
    for _ in range(5):
        solver.step()
    assert solver.nParticles == nBefore


def testHybridBoundaryEnforcement() -> None:
    """PIC/FLIP particles stay within grid bounds."""
    solver = HybridSolver(gridWidth=48, gridHeight=24, nParticlesPerRow=10)
    for _ in range(20):
        solver.step()
    pos = solver.getParticlePositions()
    assert np.all(pos[:, 0] >= 1.0)
    assert np.all(pos[:, 0] <= solver.gridWidth - 1.0)
    assert np.all(pos[:, 1] >= 1.0)
    assert np.all(pos[:, 1] <= solver.gridHeight - 1.0)
