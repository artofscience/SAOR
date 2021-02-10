from abc import ABC, abstractmethod


class Response(ABC):
    """ Abstract implementation of a response function.
        The class should provide the `self.f` and `self.df` methods to return the
        function's value and the derivative at the provided variable.
    """

    def __init__(self, n, m):
        """ Dimensionality of the problem, e.g. length of variables `x` and constraints `g`. """
        self.n = n
        self.m = m

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
