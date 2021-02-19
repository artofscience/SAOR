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
        return 1.0

    def ddy(self, x):
        return 0.0


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

    def update_intervening(self, df, **kwargs):
        self.positive = df > 0
        self.negative = df < 0

    def y(self, x):
        y = np.zeros_like(x)
        y[self.positive] = self.lin.y(x[self.positive])
        y[self.negative] = self.rec.y(x[self.negative])
        return y

    def dy(self, x):
        dy = np.zeros_like(x)
        dy[self.positive] = self.lin.dy(x[self.positive])
        dy[self.negative] = self.rec.dy(x[self.negative])
        return dy

    def ddy(self, x):
        ddy = np.zeros_like(x)
        ddy[self.positive] = self.lin.ddy(x[self.positive])
        ddy[self.negative] = self.rec.ddy(x[self.negative])
        return ddy


class MMA(Intervening):

    def __init__(self, asyinit=0.5):
        self.L = None
        self.U = None

        self.factor = asyinit
        self.asyinit = asyinit
        self.asyincr = 1.1
        self.asydecr = 0.7
        self.x = None
        self.xold1, self.xold2 = None, None


    def update_intervening(self, x, f, df, xmin, xmax, **kwargs):
        self.xold2 = self.xold1
        self.xold1 = self.x
        self.x = x

        self.dx = xmax - xmin

        # Initial values of asymptotes
        if self.xold1 is not None and self.xold2 is not None:
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