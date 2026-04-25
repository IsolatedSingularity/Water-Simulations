"""Tests for SPH kernel functions: normalization and gradient properties."""

import numpy as np
import pytest

from watersim.solvers.kernels import poly6Kernel, spikyGradKernel, viscLaplacianKernel


def testPoly6NonNegative() -> None:
    """poly6Kernel returns non-negative values for all inputs."""
    h = 1.0
    rSq = np.linspace(0, h**2 + 0.1, 50)
    result = poly6Kernel(rSq, h)
    assert np.all(result >= 0.0), "poly6Kernel produced negative values."


def testPoly6ZeroOutsideSupport() -> None:
    """poly6Kernel is exactly zero when rSq >= h^2."""
    h = 0.8
    rSq = np.array([h**2, h**2 + 0.01, 2.0])
    result = poly6Kernel(rSq, h)
    assert np.all(result == 0.0), "poly6Kernel nonzero outside support."


def testPoly6MaxAtOrigin() -> None:
    """poly6Kernel is maximized at r=0."""
    h = 0.8
    rSqVals = np.array([0.0, 0.1, 0.3, 0.5, h**2 - 1e-6])
    vals = poly6Kernel(rSqVals, h)
    assert vals[0] == pytest.approx(max(vals), rel=1e-6), "poly6Kernel not maximized at origin."


def testSpikyGradZeroAtOrigin() -> None:
    """spikyGradKernel returns zero gradient at r=0 (no singular force)."""
    h = 0.8
    rVec = np.array([[0.0, 0.0]])
    r = np.array([0.0])
    result = spikyGradKernel(rVec, r, h)
    assert np.allclose(result, 0.0), "spikyGradKernel nonzero at r=0."


def testSpikyGradZeroOutsideSupport() -> None:
    """spikyGradKernel is zero outside support (r >= h)."""
    h = 0.8
    rVec = np.array([[h + 0.01, 0.0], [1.0, 1.0]])
    r = np.linalg.norm(rVec, axis=1)
    result = spikyGradKernel(rVec, r, h)
    assert np.allclose(result, 0.0), "spikyGradKernel nonzero outside support."


def testViscLaplacianNonNegative() -> None:
    """viscLaplacianKernel returns non-negative values inside support."""
    h = 0.8
    r = np.linspace(0, h - 1e-9, 20)
    result = viscLaplacianKernel(r, h)
    assert np.all(result >= 0.0), "viscLaplacianKernel negative inside support."


def testViscLaplacianZeroOutsideSupport() -> None:
    """viscLaplacianKernel is zero at and beyond h."""
    h = 0.8
    r = np.array([h, h + 0.01, 1.5])
    result = viscLaplacianKernel(r, h)
    assert np.all(result == 0.0), "viscLaplacianKernel nonzero outside support."


def testPoly6SymmetryInRSq() -> None:
    """poly6Kernel depends only on rSq, not direction."""
    h = 1.0
    rSq1 = np.array([0.25])
    rSq2 = np.array([0.25])
    assert poly6Kernel(rSq1, h)[0] == pytest.approx(poly6Kernel(rSq2, h)[0])
