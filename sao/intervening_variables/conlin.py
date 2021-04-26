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
        y[self.positive] = np.broadcast_to(self.lin.y(x), self.positive.sum())
        y[~self.positive] = np.broadcast_to(self.rec.y(x), (~self.positive).sum())
        return y

    def dydx(self, x):
        dydx = np.zeros_like(self.positive, dtype=float)
        dydx[self.positive] = np.broadcast_to(self.lin.dydx(x), self.positive.sum())
        dydx[~self.positive] = np.broadcast_to(self.rec.dydx(x), (~self.positive).sum())
        return dydx

    def ddyddx(self, x):
        ddyddx = np.zeros_like(self.positive, dtype=float)
        ddyddx[self.positive] = np.broadcast_to(self.lin.ddyddx(x), self.positive.sum())
        ddyddx[~self.positive] = np.broadcast_to(self.rec.ddyddx(x), (~self.positive).sum())
        return ddyddx

    def dxdy(self, x):
        dxdy = np.zeros_like(self.positive, dtype=float)
        dxdy[self.positive] = np.broadcast_to(self.lin.dxdy(x), self.positive.sum())
        dxdy[~self.positive] = np.broadcast_to(self.rec.dxdy(x), (~self.positive).sum())
        return dxdy

    def ddxddy(self, x):
        ddxddy = np.zeros_like(self.positive, dtype=float)
        ddxddy[self.positive] = np.broadcast_to(self.lin.ddxddy(x), self.positive.sum())
        ddxddy[~self.positive] = np.broadcast_to(self.rec.ddxddy(x), (~self.positive).sum())
        return ddxddy
