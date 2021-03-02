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

    def __init__(self, asyinit=0.5, globbound = (0, 1)):
        self.L = None
        self.U = None

        self.factor = asyinit
        self.asyinit = asyinit
        self.asyincr = 1.1
        self.asydecr = 0.7
        self.x = None
        self.xold1, self.xold2 = None, None
        self.dx = globbound[1] - globbound[0]


    def update_intervening(self, x, f, df, xmin, xmax, **kwargs):
        self.xold2 = self.xold1
        self.xold1 = self.x
        self.x = x

        # Initial values of asymptotes
        if self.xold2 is not None:
            self.low = self.x - self.factor * self.dx
            self.upp = self.x + self.factor * self.dx

        # Update asymptotes
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

            # minimum variable bounds
            zzl1 = self.low + self.albefa * (self.x - self.low)  # limit change in x_i wrt asymptotes U_i, L_i
            zzl2 = self.x - self.move_limit * self.dx
            xmin[:] = np.maximum.reduce([zzl1, zzl2, xmin])  # finds the max for each row of (zzl1, zzl2, xmin)

            # maximum variable bounds
            zzu1 = self.upp - self.albefa * (self.upp - self.x)  # limit change in x_i wrt asymptotes U_i, L_i
            zzu2 = self.x + self.move_limit * self.dx
            self.beta = np.minimum.reduce([zzu1, zzu2, self.xmax])  # finds the min for each row of (zzu1, zzu2, xmax)

            xmin[:] = 1.1 * xmin