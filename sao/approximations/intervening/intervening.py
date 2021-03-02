from abc import ABC, abstractmethod
import numpy as np


class Intervening(ABC):
    def update_intervening(self, **kwargs):
        pass

    @abstractmethod
    def y(self, x):
        ...

    @abstractmethod
    def dy(self, x):
        ...

    @abstractmethod
    def ddy(self, x):
        ...


class Linear(Intervening):
    def y(self, x):
        return x

    def dy(self, x):
        return np.ones_like(x)

    def ddy(self, x):
        return np.zeros_like(x)


class Reciprocal(Intervening):
    def y(self, x):
        return 1 / x

    def dy(self, x):
        return -1 / (x ** 2)

    def ddy(self, x):
        return 2 / (x ** 3)


class ConLin(Intervening):
    def __init__(self):
        self.positive = None
        self.negative = None
        self.lin = Linear()
        self.rec = Reciprocal()

    def update_intervening(self, dg, **kwargs):
        self.positive = dg > 0
        self.negative = dg < 0

    def y(self, x):
        y = np.zeros_like(self.positive, dtype=float)
        y[self.positive] = self.lin.y(np.broadcast_to(x, self.positive.shape)[self.positive])
        y[self.negative] = self.rec.y(np.broadcast_to(x, self.negative.shape)[self.negative])
        return y

    def dy(self, x):
        dy = np.zeros_like(self.positive, dtype=float)
        dy[self.positive] = self.lin.dy(np.broadcast_to(x, self.positive.shape)[self.positive])
        dy[self.negative] = self.rec.dy(np.broadcast_to(x, self.negative.shape)[self.negative])
        return dy

    def ddy(self, x):
        ddy = np.zeros_like(self.positive, dtype=float)
        ddy[self.positive] = self.lin.ddy(np.broadcast_to(x, self.positive.shape)[self.positive])
        ddy[self.negative] = self.rec.ddy(np.broadcast_to(x, self.negative.shape)[self.negative])
        return ddy


class MMA(Intervening):

    def __init__(self, xmin, xmax):
        self.x = None
        self.xold1, self.xold2 = None, None
        self.low, self.upp = None, None
        self.positive = None
        self.negative = None
        self.asyinit = 0.5
        self.asyincr = 1.1
        self.asydecr = 0.7
        self.asybound = 10.0
        self.albefa = 0.1               # limit the max change of vars wrt asymptotes
        self.factor = self.asyinit * np.ones(len(xmin))
        self.dx = xmax - xmin

    def update_intervening(self, x, g, dg, **kwargs):
        self.xold2 = self.xold1
        self.xold1 = self.x
        self.x = x
        self.positive = dg > 0   # size of [m_p, n_l]
        self.negative = dg < 0   # size of [m_p, n_l]
        self.get_asymptotes()

    def get_asymptotes(self):

        # Initial values of asymptotes
        if self.xold2 is not None:
            self.low = self.x - self.factor * self.dx
            self.upp = self.x + self.factor * self.dx

        # Update scheme for asymptotes
        else:
            # depending on if the signs of (x_k-xold) and (xold-xold2) are opposite, indicating an oscillation in xi
            # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed

            # check for oscillations in variables (if zzz > 0: no oscillations, if zzz < 0: oscillations)
            zzz = (self.x - self.xold1) * (self.xold1 - self.xold2)

            # oscillating variables x_i are assigned a factor of asydecr and non-oscillating to asyincr
            self.factor[zzz > 0] = self.asyincr
            self.factor[zzz < 0] = self.asydecr

            # update lower and upper asymptotes
            self.low = self.x - self.factor * (self.xold1 - self.low)
            self.upp = self.x + self.factor * (self.upp - self.xold1)

            # check min and max bounds of asymptotes, as they cannot be too close or far from the variable (redundant?)
            lowmin = self.x - self.asybound * self.dx
            lowmax = self.x - 1 / (self.asybound ** 2) * self.dx
            uppmin = self.x + 1 / (self.asybound ** 2) * self.dx
            uppmax = self.x + self.asybound * self.dx

            # if given asymptotes cross boundaries put them to their max/min values (redundant?)
            self.low = np.maximum(self.low, lowmin)
            self.low = np.minimum(self.low, lowmax)
            self.upp = np.minimum(self.upp, uppmax)
            self.upp = np.maximum(self.upp, uppmin)

    def get_bounds(self):
        zzl1 = self.low + self.albefa * (self.x - self.low)
        zzu1 = self.upp - self.albefa * (self.upp - self.x)
        return zzl1, zzu1

    # TODO: Below is how I implemented the intervening vars for MMA. Needs to be edited to fit the current framework
    def y(self, x):
        y = np.zeros(self.positive.shape, dtype=float)
        y[self.positive] = (1 / (self.upp - x))[self.positive]          # TODO: shapes do not match
        y[self.negative] = 1 / (x - self.low)[self.negative]
        return y

    def dy(self, x):
        dy = np.zeros(self.positive.shape, dtype=float)
        for j in range(0, self.positive.shape[1]):
            dy[self.positive] = (1 / (self.upp - x) ** 2)[self.positive]
            dy[self.negative] = (-1 / (x - self.low) ** 2)[self.negative]
        return dy

    def ddy(self, x):
        ddy = np.zeros(self.positive.shape, dtype=float)
        for j in range(0, self.positive.shape[1]):
            ddy[self.dg[j, :] >= 0, j] = (2 / (self.upp - x) ** 3)[self.dg[j, :] >= 0]
            ddy[self.dg[j, :] < 0, j] = (2 / (x - self.low) ** 3)[self.dg[j, :] < 0]
        return ddy
