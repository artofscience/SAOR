from .intervening import Intervening
import numpy as np

class Exponential(Intervening):
    """A generic exponential intervening variable y = x^p.

    The general case for an exponential intervening varaibles that can take
    on various forms depending on the chosen power. Note: the implementation
    does not support ``p = 0`` to avoid a zero devision in the derivatives.
    """

    def __init__(self, p):
        """Initialise the exponential intervening variable with a power."""
        assert p != 0, f"Invalid power x^{p}, will result in zero division."
        self.p = p

    def y(self, x):
        return x ** self.p

    def dydx(self, x):
        return self.p * x ** (self.p - 1)

    def ddyddx(self, x):
        return self.p * (self.p - 1) * x ** (self.p - 2)

    def clip(self, x):  # TODO Maybe add some tolerance like albefa to keep the value from 0
        if self.p < 0:
            return np.maximum(x, 0.0, out=x)
        else:
            return x


class Linear(Exponential):
    """A linear intervening variable: ``y = x``."""
    def __init__(self):
        super().__init__(1)


class Reciprocal(Exponential):
    """A reciprocal intervening variable: ``y = 1/x``."""
    def __init__(self):
        super().__init__(-1)


class ReciSquared(Exponential):
    """A squared reciprocal intervening variable: ``y = 1/x^2``."""
    def __init__(self):
        super().__init__(-2)


class ReciCubed(Exponential):
    """A cubic reciprocal intervening variable: ``y = 1/x^3``."""
    def __init__(self):
        super().__init__(-3)
