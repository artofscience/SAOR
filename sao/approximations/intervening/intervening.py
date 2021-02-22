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


class Linear(Intervening):          # TODO: Make the shape of -y- consistent with ConLin and MMA
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
        y = np.zeros_like(x)            # TODO: self.positive is a matrix of [m+1, n], while y is a vector of [n,]
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

    # TODO: Imo we might want to have the bounds in vector form to allow for different bounds per variable set
    def __init__(self, asyinit=0.5, globbound=(0, 1)):
        self.x = None
        self.xold1, self.xold2 = None, None
        self.low, self.upp = None, None
        self.positive = None
        self.negative = None

        # TODO: You need -n- here. We can extract it from the bounds if we pass them as vectors
        self.factor = asyinit * np.ones(self.n)

        self.asyinit = asyinit
        self.asyincr = 1.1
        self.asydecr = 0.7
        self.asybound = 10.0
        self.albefa = 0.1               # albefa is limiting the max change of vars wrt asymptotes

        self.dx = globbound[1] - globbound[0]

    def update_intervening(self, x, f, df, xmin, xmax, **kwargs):
        self.xold2 = self.xold1
        self.xold1 = self.x
        self.x = x
        self.positive = df > 0
        self.negative = df < 0
        # self.dx = xmax - xmin

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

            # TODO: How should we handle the local sub-problem bounds alpha, beta?
            #  Imo they should be computed by a separate method. Not sure within which class though, as they depend on
            #  the move_limit and the intervening variables (e.g. asymptotes). I would place it in the
            #  InterveningApproximation class because all approximations generate local sub-problem bounds and
            #  this class has access to the intervening vars (i.e. self.iter).

            # minimum variable bounds
            zzl1 = self.low + self.albefa * (self.x - self.low)  # limit change in x_i wrt asymptotes U_i, L_i
            zzl2 = self.x - self.move_limit * self.dx
            xmin[:] = np.maximum.reduce([zzl1, zzl2, xmin])  # finds the max for each row of (zzl1, zzl2, xmin)

            # maximum variable bounds
            zzu1 = self.upp - self.albefa * (self.upp - self.x)  # limit change in x_i wrt asymptotes U_i, L_i
            zzu2 = self.x + self.move_limit * self.dx
            self.beta = np.minimum.reduce([zzu1, zzu2, self.xmax])  # finds the min for each row of (zzu1, zzu2, xmax)

            xmin[:] = 1.1 * xmin

    # TODO: Below is how I implemented the intervening vars for MMA. Not sure how it can be implemented
    #  more efficiently, just keep in mind that these methods are called within the solver everytime you want to
    #  evaluate g_approx, dg_approx and ddg_approx at a different point -x-, so multiple times per design iteration.
    def y(self, x):
        y = np.empty((self.n, self.m + 1))
        for j in range(0, self.m + 1):
            y[self.dg[j, :] >= 0, j] = (1 / (self.upp - x))[self.dg[j, :] >= 0]
            y[self.dg[j, :] < 0, j] = 1 / (x - self.low)[self.dg[j, :] < 0]
        return y

    def dy(self, x):
        dy = np.empty((self.n, self.m + 1))
        for j in range(0, self.m + 1):
            dy[self.dg[j, :] >= 0, j] = (1 / (self.upp - x) ** 2)[self.dg[j, :] >= 0]
            dy[self.dg[j, :] < 0, j] = (-1 / (x - self.low) ** 2)[self.dg[j, :] < 0]
        return dy

    def ddy(self, x):
        ddy = np.empty((self.n, self.m + 1))
        for j in range(0, self.m + 1):
            ddy[self.dg[j, :] >= 0, j] = (2 / (self.upp - x) ** 3)[self.dg[j, :] >= 0]
            ddy[self.dg[j, :] < 0, j] = (2 / (x - self.low) ** 3)[self.dg[j, :] < 0]
        return ddy
