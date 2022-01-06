import numpy as np
from .mapping import Mapping

class Exponential(Mapping):
    """A generic exponential intervening variable y = x^p.

    The general case for an exponential intervening varaibles that can take
    on various forms depending on the chosen power. Note: the implementation
    does not support ``p = 0`` to avoid a zero devision in the derivatives.
    """

    def __init__(self, p, xlim=1e-10):
        """
        Initialise the exponential intervening variable with a power.
        :param p: The power
        :param xlim: Minimum x, in case of negative p, to prevent division by 0
        """
        assert p != 0, f"Invalid power x^{p}, will result in zero division."
        self.p = p
        self.xlim = xlim

    def g(self, x):
        return x ** self.p

    def dg(self, x):
        return self.p * x ** (self.p - 1)

    def ddg(self, x):
        return self.p * (self.p - 1) * x ** (self.p - 2)

    def clip(self, x):
        if self.p < 0:
            return np.maximum(x, self.xlim, out=x)
        return x
