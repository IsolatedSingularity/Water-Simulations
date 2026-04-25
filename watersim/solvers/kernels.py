"""SPH kernel functions: poly6, spiky gradient, viscosity Laplacian."""

import numpy as np


def poly6Kernel(rSq: np.ndarray, h: float) -> np.ndarray:
    """
    Poly6 smoothing kernel (scalar field estimation).

    Parameters
    ----------
    rSq: squared distances, shape (N,).
    h:   smoothing radius.

    Returns
    -------
    Kernel weights, shape (N,). Zero outside support.
    """
    hSq = h * h
    coeff = 315.0 / (64.0 * np.pi * h**9)
    mask = rSq < hSq
    result = np.zeros_like(rSq)
    result[mask] = coeff * (hSq - rSq[mask]) ** 3
    return result


def spikyGradKernel(rVec: np.ndarray, r: np.ndarray, h: float) -> np.ndarray:
    """
    Spiky gradient kernel (pressure force).

    Parameters
    ----------
    rVec: displacement vectors, shape (N, 2).
    r:    distances, shape (N,).
    h:    smoothing radius.

    Returns
    -------
    Gradient vectors, shape (N, 2). Zero outside support or at r=0.
    """
    coeff = -45.0 / (np.pi * h**6)
    mask = (r > 1e-9) & (r < h)
    result = np.zeros_like(rVec)
    if np.any(mask):
        r_safe = r[mask]
        scalars = (coeff * (h - r_safe) ** 2 / r_safe)[:, np.newaxis]
        result[mask] = scalars * rVec[mask]
    return result


def viscLaplacianKernel(r: np.ndarray, h: float) -> np.ndarray:
    """
    Viscosity Laplacian kernel (viscous force).

    Parameters
    ----------
    r: distances, shape (N,).
    h: smoothing radius.

    Returns
    -------
    Scalar Laplacian weights, shape (N,). Zero outside support.
    """
    coeff = 45.0 / (np.pi * h**6)
    mask = r < h
    result = np.zeros_like(r)
    result[mask] = coeff * (h - r[mask])
    return result
