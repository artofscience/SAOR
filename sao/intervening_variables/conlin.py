import numpy as np

from .intervening import Intervening
from .exponential import Linear, Reciprocal


class ConLin(Intervening):
    """The ``ConLin`` formulation of intervening variables.

    A combination of `Linear` and `Reciprocal` intervening variables.
    # TODO: improve documentation including references.
    """
    def __init__(self):
        # The linear and reciprocal intervening variables
        self.lin = Linear()
        self.rec = Reciprocal()

        # A boolean indicator array that keeps track of the positive (and
        # negative) values of the variables. This is used to switch the
        # Linear and Reciprocal intervening variables.
        self.positive = None

    def update(self, x, f, df):
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

    def dxdy(self, x):
        dxdy = np.zeros_like(self.positive, dtype=float)
        for (row, positive) in enumerate(self.positive):
            dxdy[row, positive] = self.lin.dxdy(x[positive])
            dxdy[row, ~positive] = self.rec.dxdy(x[~positive])
        return dxdy

    def ddxddy(self, x):
        ddxddy = np.zeros_like(self.positive, dtype=float)
        for (row, positive) in enumerate(self.positive):
            ddxddy[row, positive] = self.lin.ddxddy(x[positive])
            ddxddy[row, ~positive] = self.rec.ddxddy(x[~positive])
        return ddxddy
