from abc import ABC, abstractmethod


class Response(ABC):
    """Abstract implementation of a response function.

    The class should provide the `self.f` and `self.df` methods to return the
    function's value and the derivative at the provided variable.
    """

    def __init__(self, n):
        # Dimensionality of the problem, e.g. length of variables `x`.
        self.n = n

    @abstractmethod
    def f(self, x):
        """Returns the function's value at `x`."""
        ...

    @abstractmethod
    def df(self, x):
        """Returns the first order derivative at `x`."""
        ...

    def ddf(self, x):
        """Optional implementation for second order derivative at `x`."""
        pass
