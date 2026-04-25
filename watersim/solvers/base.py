"""Abstract base class for all fluid solvers."""

from abc import ABC, abstractmethod


class Solver(ABC):
    """Abstract base: every solver must implement step()."""

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one time step."""
        ...
