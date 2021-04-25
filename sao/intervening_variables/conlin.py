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
        y[self.positive] = self.lin.y(np.broadcast_to(x, self.positive.shape)[self.positive])
        y[~self.positive] = self.rec.y(np.broadcast_to(x, self.positive.shape)[~self.positive])
        return y

    def dydx(self, x):
        dydx = np.zeros_like(self.positive, dtype=float)
        dydx[self.positive] = self.lin.dydx(np.broadcast_to(x, self.positive.shape)[self.positive])
        dydx[~self.positive] = self.rec.dydx(np.broadcast_to(x, self.positive.shape)[~self.positive])
        return dydx

    def ddyddx(self, x):
        ddyddx = np.zeros_like(self.positive, dtype=float)
        ddyddx[self.positive] = self.lin.ddyddx(np.broadcast_to(x, self.positive.shape)[self.positive])
        ddyddx[~self.positive] = self.rec.ddyddx(np.broadcast_to(x, self.positive.shape)[~self.positive])
        return ddyddx

    def dxdy(self, x):
        dxdy = np.zeros_like(self.positive, dtype=float)
        dxdy[self.positive] = self.lin.dxdy(np.broadcast_to(x, self.positive.shape)[self.positive])
        dxdy[~self.positive] = self.rec.dxdy(np.broadcast_to(x, self.positive.shape)[~self.positive])
        return dxdy

    def ddxddy(self, x):
        ddxddy = np.zeros_like(self.positive, dtype=float)
        ddxddy[self.positive] = self.lin.ddxddy(np.broadcast_to(x, self.positive.shape)[self.positive])
        ddxddy[~self.positive] = self.rec.ddxddy(np.broadcast_to(x, self.positive.shape)[~self.positive])
        return ddxddy
