import numpy as np

from .exponential import Linear, Reciprocal
from .intervening import Intervening


class PositiveNegative(Intervening):
    def __init__(self, left: Intervening, right: Intervening):
        self.left = left
        self.right = right
        self.positive = None

    def update(self, x, f, df, *args, **kwargs):
        self.positive = df >= 0

    def y(self, x):
        return np.where(self.positive, self.right.y(x), self.left.y(x))

    def dydx(self, x):
        return np.where(self.positive, self.right.dydx(x), self.left.dydx(x))

    def ddyddx(self, x):
        return np.where(self.positive, self.right.ddyddx(x), self.left.ddyddx(x))

    def clip(self, x):
        self.left.clip(x)
        self.right.clip(x)
        return x


class ConLin(PositiveNegative):
    """The ``ConLin`` formulation of intervening variables.

    Introduced in: https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620230307.
    This is a 1st-order Taylor expansion with respect to the mixed
    (i.e. a combination of ``Linear`` and ``Reciprocal``)

    Intervening variables:

    .. math::
        y_i = x_i \\quad&\\text{if}\\quad dg_j/dx_i >= 0 \\\\
        y_i = \\frac{1}{x_i} \\quad&\\text{if}\\quad dg_j/dx_i < 0
    """

    def __init__(self):
        super().__init__(Reciprocal(), Linear())

    def clip(self, x):  # TODO Maybe add some tolerance like albefa to keep the value from 0
        return np.maximum(x, 0.0, out=x)
