from abc import ABC, abstractmethod


class Problem(ABC):
    """ Abstract implementation of problem.
        The class should provide the response functions `g`, sensitivities `dg` and (possibly) 2nd-order
        sensitivities `ddg` at a given design point `x`.
    """

    def __init__(self):
        pass

    @abstractmethod
    def g(self, x):
        """Returns the function's value at `x`."""
        return NotImplemented

    @abstractmethod
    def dg(self, x):
        """Returns the 1st-order derivatives at `x`."""
        return NotImplemented

    def ddg(self, x):
        """Optional implementation for diagonal 2nd-order derivatives at `x`."""
        return NotImplemented
