from abc import ABC, abstractmethod
import numpy as np


class Intervening(ABC):
    def update(self, x, f, df):
        pass

    @abstractmethod
    def y(self, x):
        ...

    @abstractmethod
    def dydx(self, x):
        ...

    @abstractmethod
    def ddyddx(self, x):
        ...

    @abstractmethod
    def dxdy(self, x):
        ...

    @abstractmethod
    def ddxddy(self, x):
        ...


class Linear(Intervening):
    def y(self, x):
        return x

    def dydx(self, x):
        return np.ones_like(x)

    def ddyddx(self, x):
        return np.zeros_like(x)

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        return np.ones_like(x)

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x):
        return np.zeros_like(x)


class Reciprocal(Intervening):
    def y(self, x):
        return 1 / x

    def dydx(self, x):
        return -1 / (x ** 2)

    def ddyddx(self, x):
        return 2 / (x ** 3)

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        return - x**2

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x):
        return 2 * x**3


class ConLin(Intervening):
    def __init__(self):
        self.positive = None
        self.lin = Linear()
        self.rec = Reciprocal()

    def update(self, x, f, df):
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

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        dxdy = np.zeros_like(self.positive, dtype=float)
        dxdy[self.positive] = self.lin.dxdy(np.broadcast_to(x, self.positive.shape)[self.positive])
        dxdy[~self.positive] = self.rec.dxdy(np.broadcast_to(x, self.positive.shape)[~self.positive])
        return dxdy

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x):
        ddxddy = np.zeros_like(self.positive, dtype=float)
        ddxddy[self.positive] = self.lin.ddxddy(np.broadcast_to(x, self.positive.shape)[self.positive])
        ddxddy[~self.positive] = self.rec.ddxddy(np.broadcast_to(x, self.positive.shape)[~self.positive])
        return ddxddy


class MMA(Intervening):

    def __init__(self, xmin, xmax, **kwargs):
        self.x = None
        self.xold1, self.xold2 = None, None
        self.low, self.upp = None, None
        self.positive = None
        self.asyinit = kwargs.get('asyinit', 0.5)
        self.asyincr = kwargs.get('asyincr', 1.1)
        self.asydecr = kwargs.get('asydecr', 0.7)
        self.asybound = kwargs.get('asydecr', 10.0)
        self.albefa = kwargs.get('asydecr', 0.1)               # limit the max change of vars wrt asymptotes
        self.factor = self.asyinit * np.ones(len(xmin))
        self.dx = xmax - xmin

    def update(self, x, f, df):
        self.xold2 = self.xold1
        self.xold1 = self.x
        self.x = x
        self.positive = df >= 0         # size of [m_p, n_l]
        self.get_asymptotes()

    def get_asymptotes(self):

        # Initial values of asymptotes
        if self.xold2 is None:
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

    def y(self, x):
        y = np.zeros_like(self.positive, dtype=float)
        y[self.positive] = np.broadcast_to((1 / (self.upp - x)), self.positive.shape)[self.positive]
        y[~self.positive] = np.broadcast_to((1 / (x - self.low)), self.positive.shape)[~self.positive]
        return y

    def dydx(self, x):
        dydx = np.zeros_like(self.positive, dtype=float)
        dydx[self.positive] = np.broadcast_to((1 / (self.upp - x)**2), self.positive.shape)[self.positive]
        dydx[~self.positive] = np.broadcast_to((-1 / (x - self.low)**2), self.positive.shape)[~self.positive]
        return dydx

    def ddyddx(self, x):
        ddyddx = np.zeros_like(self.positive, dtype=float)
        ddyddx[self.positive] = np.broadcast_to((2 / (self.upp - x) ** 3), self.positive.shape)[self.positive]
        ddyddx[~self.positive] = np.broadcast_to((2 / (x - self.low) ** 3), self.positive.shape)[~self.positive]
        return ddyddx

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        dxdy = np.zeros_like(self.positive, dtype=float)
        dxdy[self.positive] = np.broadcast_to((1 / self.y(x) ** 2), self.positive.shape)[self.positive]
        dxdy[~self.positive] = np.broadcast_to((-1 / self.y(x) ** 2), self.positive.shape)[~self.positive]
        return dxdy

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x, **kwargs):
        ddxddy = np.zeros_like(self.positive, dtype=float)
        ddxddy[self.positive] = np.broadcast_to((-2 / self.y(x) ** 3), self.positive.shape)[self.positive]
        ddxddy[~self.positive] = np.broadcast_to((2 / self.y(x) ** 3), self.positive.shape)[~self.positive]
        return ddxddy

    def get_move_limit(self):
        zzl1 = self.low + self.albefa * (self.x - self.low)
        zzu1 = self.upp - self.albefa * (self.upp - self.x)
        return zzl1, zzu1


class ReciSquared(Intervening):
    def y(self, x):
        return 1 / x**2

    def dydx(self, x):
        return -2 / (x ** 3)

    def ddyddx(self, x):
        return 6 / (x ** 4)

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        return - 0.5 * x**3

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x):
        return 3/4 * x**5


class ReciCubed(Intervening):
    def y(self, x):
        return 1 / x**3

    def dydx(self, x):
        return -3 / (x ** 4)

    def ddyddx(self, x):
        return 12 / (x ** 5)

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        return - 1/3 * x**4

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x):
        return 3/9 * x**7
