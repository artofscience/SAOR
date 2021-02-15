from abc import ABC, abstractmethod


class Problem(ABC):
    """ Abstract implementation of problem.
        The class should provide the response functions `g`, sensitivities `dg` and (possibly) 2nd-order
        sensitivities `ddg` at a given design point `x`.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_g(self, x):
        """Calculates the function's value at `x`."""
        pass

    @abstractmethod
    def get_dg(self, x):
        """Calculates the 1st-order derivatives at `x`."""
        pass

    def get_ddg(self, x):
        """Optional implementation for diagonal 2nd-order derivatives at `x`."""
        pass
