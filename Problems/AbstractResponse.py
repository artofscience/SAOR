from abc import ABC, abstractmethod


class Response(ABC):
    """ Abstract implementation of a response function.
        The class should provide the `self.g` and `self.dg` methods to return the
        response function's value and its derivatives at a given design point `x`.
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
