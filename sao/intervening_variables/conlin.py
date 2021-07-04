import numpy as np

from .intervening import Intervening
from .exponential import Linear, Reciprocal


class ConLin(Intervening):
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
        # The linear and reciprocal intervening variables
        self.lin = Linear()
        self.rec = Reciprocal()

        # A boolean indicator array that keeps track of the positive (and
        # negative) values of the variables. This is used to switch the
        # Linear and Reciprocal intervening variables.
        self.positive = None

    def update(self, x, f, df, *args, **kwargs):
        """Update the boolean indicator of the positive entries."""
        self.positive = df >= 0

    def y(self, x):
        y = np.zeros_like(self.positive, dtype=float)
        for (row, positive) in enumerate(self.positive):
            y[row, positive] = self.lin.y(x[positive])
            y[row, ~positive] = self.rec.y(x[~positive])
        return y

    def dydx(self, x):
        dydx = np.zeros_like(self.positive, dtype=float)
        for (row, positive) in enumerate(self.positive):
            dydx[row, positive] = self.lin.dydx(x[positive])
            dydx[row, ~positive] = self.rec.dydx(x[~positive])
        return dydx

    def ddyddx(self, x):
        ddyddx = np.zeros_like(self.positive, dtype=float)
        for (row, positive) in enumerate(self.positive):
            ddyddx[row, positive] = self.lin.ddyddx(x[positive])
            ddyddx[row, ~positive] = self.rec.ddyddx(x[~positive])
        return ddyddx

    def clip(self, x):  # TODO Maybe add some tolerance like albefa to keep the value from 0
        return np.maximum(x, 0.0, out=x)
