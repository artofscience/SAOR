from dataclasses import dataclass, fields
import numpy as np


@dataclass
class State(object):
    """A dataclass carrying the state for the IPOpt solvers.

    This dataclass contains the often used variables throughout the IP solvers.
    Additionally, some basic functionality is provided, such as addition
    and multiplication. For addition, two states can be added where the values
    of the corresponding fields are summed together. For multiplication, it is
    assumed to be multiplied with a single scalar value, scaling_strategies all fields
    within the state object.

    TODO: document the variables and what they represent
    """
    x: np.array
    xsi: float
    eta: float
    lam: np.array
    s: np.array

    def __iter__(self):
        """Iterator over all values of each field present in the state."""
        for field in fields(self):
            yield getattr(self, field.name)

    def __add__(self, other):
        """Addition of two states, where each field is summed together."""
        return self.__class__(*(s + o for s, o in zip(self, other)))

    def __mul__(self, other):
        """Multiplication with a scalar value to scale all fields."""
        return self.__class__(*(value * other for value in self))

    def __rmul__(self, other):
        """Right multiplication with a scalar, see ``__mul__``."""
        return self.__mul__(other)

    def norm(self):
        """Return the norm of all stacked variables."""
        return np.linalg.norm(np.hstack(tuple(self)))

    def max(self):
        """Return the abs maximum of all stacked variables in self."""
        return np.max(np.abs(np.hstack(tuple(self))))


@dataclass
class StateY(State):
    """An extended data class adding the ``y`` and ``mu`` variables."""
    y: np.array
    mu: float


@dataclass
class StateYZ(StateY):
    """An extended data class adding the ``z`` and ``zeta`` variables."""
    z: np.array
    zeta: np.array
