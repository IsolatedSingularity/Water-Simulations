"""Fluid dynamics solvers: SPH, Stable Fluids, PIC/FLIP."""

from watersim.solvers.sph import SPHSolver
from watersim.solvers.stableFluids import StableFluidsSolver
from watersim.solvers.picFlip import HybridSolver

__all__ = ["SPHSolver", "StableFluidsSolver", "HybridSolver"]
