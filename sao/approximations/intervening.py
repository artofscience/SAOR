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
        return - x ** 2

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x):
        return 2 * x ** 3


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


class MMASquared(MMA):
    def y(self, x):
        y = np.zeros_like(self.positive, dtype=float)
        y[self.positive] = np.broadcast_to((1 / (self.upp - x)**2), self.positive.shape)[self.positive]
        y[~self.positive] = np.broadcast_to((1 / (x - self.low)**2), self.positive.shape)[~self.positive]
        return y

    def dydx(self, x):
        dydx = np.zeros_like(self.positive, dtype=float)
        dydx[self.positive] = np.broadcast_to((2 / (self.upp - x)**3), self.positive.shape)[self.positive]
        dydx[~self.positive] = np.broadcast_to((-2 / (x - self.low)**3), self.positive.shape)[~self.positive]
        return dydx

    def ddyddx(self, x):
        ddyddx = np.zeros_like(self.positive, dtype=float)
        ddyddx[self.positive] = np.broadcast_to((6 / (self.upp-x)**4), self.positive.shape)[self.positive]
        ddyddx[~self.positive] = np.broadcast_to((6 / (x-self.low)**4), self.positive.shape)[~self.positive]
        return ddyddx

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        dxdy = np.zeros_like(self.positive, dtype=float)
        temp1 = (self.upp - 1/self.y(x))
        temp2 = (1/self.y(x) - self.low)
        temp1[temp1 < 0] = 1e-3
        temp2[temp2 < 0] = 1e-3
        dxdy[self.positive] = np.broadcast_to((1 / (2*(self.y(x))**(3/2))), self.positive.shape)[self.positive]
        dxdy[~self.positive] = np.broadcast_to((-1 / (2*(self.y(x))**(3/2))), self.positive.shape)[~self.positive]
        return dxdy

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x, **kwargs):
        ddxddy = np.zeros_like(self.positive, dtype=float)
        ddxddy[self.positive] = np.broadcast_to((-3 / (4 * x**(5/2))), self.positive.shape)[self.positive]
        ddxddy[~self.positive] = np.broadcast_to((3 / (4 * x**(5/2))), self.positive.shape)[~self.positive]
        return ddxddy


class ReciFit(Intervening):
    def __init__(self):
        self.x = None
        self.a, self.b = None, None
        self.f, self.df, self.dfold1, self.fold1 = None, None, None, None

    def update(self, x, f, df):
        self.fold1 = self.f
        self.f = f
        self.dfold1 = self.df
        self.df = df
        self.x = x
        self.get_coefficients()

    def get_coefficients(self):
        if self.dfold1 is not None:
            self.a = - 0.5 * (self.df + self.dfold1) / (0.5 * (self.f + self.fold1)) ** 2
            self.b = 2 / (self.f + self.fold1) - self.a * self.x
        else:
            self.a = - self.df / self.f ** 2
            self.b = 1 / self.f - self.a * self.x
        self.a[(abs(self.a) < 1e-2) * (self.a >= 0)] = 1e-2
        self.a[(abs(self.a) < 1e-2) * (self.a < 0)] = -1e-2
        # self.a[abs(self.a) < 1e-5] = 1e-5 * np.sign(self.a[abs(self.a) < 1e-5])

    def y(self, x):
        y = 1 / (self.a * x + self.b)
        return y

    def dydx(self, x):
        dydx = - self.a / (self.a * x + self.b) ** 2
        return dydx

    def ddyddx(self, x):
        ddyddx = 2 * self.a ** 2 / (self.a * x + self.b) ** 3
        return ddyddx

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        dxdy = - (self.a * x + self.b) ** 2 / self.a
        return dxdy

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x):
        ddxddy = 2 * (self.a * self.x + self.b) ** 3 / self.a
        return ddxddy


class Bezier(Intervening):
    def __init__(self):
        self.x = None
        self.a, self.b, self.c = None, None, None
        self.f, self.fold1, self.fold2 = None, None, None

    def update(self, x, f, df):
        self.x = x
        self.fold2 = self.fold1
        self.fold1 = self.f
        self.f = f
        self.get_coefficients()

    def get_coefficients(self):
        if self.fold2 is None:
            self.a = 2 * self.f * np.ones_like(self.x)
        else:
            self.a = self.fold2 * np.ones_like(self.x)
        if self.fold1 is None:
            self.b = 1.5 * self.f * np.ones_like(self.x)
        else:
            self.b = self.fold1 * np.ones_like(self.x)
        self.c = self.f * np.ones_like(self.x)

    def y(self, x):
        y = (1-x)**2*self.a + 2*(1-x)*x*self.b + x**2*self.c
        return y

    def dydx(self, x):
        dydx = -2*self.a*(1-x) - 4*self.b*x + 2*self.b + 2*self.c*x
        return dydx

    def ddyddx(self, x):
        ddyddx = 2 * (self.a - 2*self.b + self.c)
        return ddyddx

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        dxdy = -1 / (2 * (-self.a*self.c + self.a*self.y(x) + self.b**2 - 2*self.b*self.y(x) + self.c*self.y(x))**(1/2))
        return dxdy

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x):
        ddxddy = (-self.a + 2*self.b - self.c) / (4*(-self.a*self.c + self.a*self.y(x) + self.b**2 - 2*self.b*self.y(x) + self.c*self.y(x))**(3/2))
        return ddxddy


class PolyFit(Intervening):
    def __init__(self):
        self.x, self.xold1 = None, None
        self.a, self.b, self.c = None, None, None
        self.f, self.fold1 = None, None
        self.df, self.dfold1 = None, None

    def update(self, x, f, df):
        if self.fold1 is None:
            self.fold1 = 2 * f
            self.dfold1 = 2 * df
            self.xold1 = 2 * x
        else:
            self.xold1 = self.x
            self.fold1 = self.f
            self.dfold1 = self.df
        self.x = x
        self.f = f
        self.df = df
        self.get_coefficients()

    def get_coefficients(self):
        # Polynomial satisfies: current point, current gradient, previous gradient
        self.a = np.zeros_like(self.x)
        bot = abs(self.x - self.xold1)
        bot2 = np.asarray(np.where(bot > 1e-4))
        bot2 = bot2[0, :]
        self.a[bot2] = (self.df[:, bot2] - self.dfold1[:, bot2]) / (2 * (self.x[bot2] - self.xold1[bot2]))
        self.a[self.a < 0] = 0
        self.b = self.df - 2 * self.a * self.x
        self.c = self.f - self.a * self.x ** 2 - self.b * self.x

    def y(self, x):
        y = self.a*x**2 + self.b*x + self.c
        return y

    def dydx(self, x):
        dydx = 2*self.a*x + self.b
        return dydx

    def ddyddx(self, x):
        ddyddx = 2*self.a
        return ddyddx

    # Define chain rule term: y = T_inv(x) --> x = T(x) --> dT/dy = dx/dy  (see ReferenceFiles/TaylorExpansion.pdf)
    def dxdy(self, x):
        # self.x1 = -self.b/(2*self.a) + (self.b**2 + 4*self.a*(self.y(x)-self.c))**(1/2) / (2*self.a)
        # self.x2 = -self.b/(2*self.a) + (self.b**2 + 4*self.a*(self.y(x)-self.c))**(1/2) / (2*self.a)
        # if 0 < self.x1 < 1:
        #     dxdy = (1 / (4*self.a*(self.y(x) - self.c) + self.b**2)**(1/2))
        # elif 0 < self.x2 < 1:
        dxdy = -1 / (4 * self.a * (self.y(x) - self.c) + self.b ** 2) ** (1 / 2)
        return dxdy

    # Define chain rule 2nd-order term: y = T_inv(x) --> x = T(x) --> d^2T/dy^2 = d^2x/dy^2  (see TaylorExpansion.pdf)
    def ddxddy(self, x):
        if 0 < self.x1 < 1:
            ddxddy = - 2*self.a / (4*self.a*(self.y(x) - self.c) + self.b**2)**(3/2)
        elif 0 < self.x2 < 1:
            ddxddy = 2 * self.a / (4 * self.a * (self.y(x) - self.c) + self.b ** 2) ** (3 / 2)
        return ddxddy

