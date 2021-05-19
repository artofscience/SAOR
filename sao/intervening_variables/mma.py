import numpy as np
from .intervening import Intervening


class MMA(Intervening):
    """The MMA algorithm, given by: http://www.ingveh.ulg.ac.be/uploads/education/meca-0027-1/MMA_DCAMM_1998.pdf

    Includes the following set of mixed intervening variables:
        y_i = 1 / (U_i - x_i)     ,  if dg_j/dx_i >= 0
        y_i = 1 / (x_i - L_i)     ,  if dg_j/dx_i < 0

    where,
        U_i := Upper asymptote (acts as an upper move-limit & adjusts the approximation's convexity)
        L_i := Lower asymptote (acts as a lower move-limit & adjusts the approximation's convexity)
    """

    def __init__(self, xmin, xmax, **kwargs):
        self.x = None
        self.xold1, self.xold2 = None, None
        self.low, self.upp = None, None

        self.xmin, self.xmax = xmin, xmax

        # MMA parameter initialization
        self.asyinit = kwargs.get('asyinit', 0.5)
        self.asyincr = kwargs.get('asyincr', 1.2)
        self.asydecr = kwargs.get('asydecr', 0.7)
        self.asybound = kwargs.get('asydecr', 10.0)
        self.albefa = kwargs.get('albefa', 0.1)
        self.factor = self.asyinit * np.ones(len(xmin))
        self.dx = xmax - xmin

        # A boolean indicator array that keeps track of the positive (and negative) values of the variables
        self.positive = None

    def update(self, x, f, df):
        """Update state of previous iterations."""
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

            # check min and max bounds of asymptotes, as they cannot be too close or far from the variable
            lowmin = self.x - self.asybound * self.dx
            lowmax = self.x - 1 / (self.asybound ** 2) * self.dx
            uppmin = self.x + 1 / (self.asybound ** 2) * self.dx
            uppmax = self.x + self.asybound * self.dx

            # if given asymptotes cross boundaries put them to their max/min values (redundant?)
            self.low = np.clip(self.low, lowmin, lowmax)
            self.upp = np.clip(self.upp, uppmin, uppmax)

        # # Fix asymptotes to constant values in order to test the influence of curvature in the responses
        # self.low = self.xmin - 0.1 * self.dx
        # self.upp = self.xmax + 0.1 * self.dx

    def y(self, x):
        y = np.zeros_like(self.positive, dtype=float)
        y[self.positive] = np.broadcast_to(1 / (self.upp - x), self.positive.shape)[self.positive]
        y[~self.positive] = np.broadcast_to(1 / (x - self.low), self.positive.shape)[~self.positive]
        return y

    def dydx(self, x):
        dydx = np.zeros_like(self.positive, dtype=float)
        dydx[self.positive] = np.broadcast_to(1 / (self.upp - x) ** 2, self.positive.shape)[self.positive]
        dydx[~self.positive] = np.broadcast_to(-1 / (x - self.low) ** 2, self.positive.shape)[~self.positive]
        return dydx

    def ddyddx(self, x):
        ddyddx = np.zeros_like(self.positive, dtype=float)
        ddyddx[self.positive] = np.broadcast_to(2 / (self.upp - x) ** 3, self.positive.shape)[self.positive]
        ddyddx[~self.positive] = np.broadcast_to(2 / (x - self.low) ** 3, self.positive.shape)[~self.positive]
        return ddyddx

    def dxdy(self, x):
        dxdy = np.zeros_like(self.positive, dtype=float)
        dxdy[self.positive] = np.broadcast_to((self.upp - x) ** 2, self.positive.shape)[self.positive]
        dxdy[~self.positive] = np.broadcast_to(-(x - self.low) ** 2, self.positive.shape)[~self.positive]
        return dxdy

    def ddxddy(self, x, **kwargs):
        ddxddy = np.zeros_like(self.positive, dtype=float)
        ddxddy[self.positive] = np.broadcast_to(-2 * (self.upp - x) ** 3, self.positive.shape)[self.positive]
        ddxddy[~self.positive] = np.broadcast_to(2 * (x - self.low) ** 3, self.positive.shape)[~self.positive]
        return ddxddy

    def get_move_limit(self):
        zzl2 = self.low + self.albefa * (self.x - self.low)
        zzu2 = self.upp - self.albefa * (self.upp - self.x)
        return zzl2, zzu2


class MMASquared(MMA):
    """A variant of the MMA intervening variables.

    Includes the following set of mixed intervening variables:
        y_i = 1 / (U_i - x_i) ** 2     ,  if dg_j/dx_i >= 0
        y_i = 1 / (x_i - L_i) ** 2     ,  if dg_j/dx_i < 0
    """

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

    def dxdy(self, x):
        dxdy = np.zeros_like(self.positive, dtype=float)
        temp1 = (self.upp - 1/self.y(x))
        temp2 = (1/self.y(x) - self.low)
        temp1[temp1 < 0] = 1e-3
        temp2[temp2 < 0] = 1e-3
        dxdy[self.positive] = np.broadcast_to((1 / (2*(self.y(x))**(3/2))), self.positive.shape)[self.positive]
        dxdy[~self.positive] = np.broadcast_to((-1 / (2*(self.y(x))**(3/2))), self.positive.shape)[~self.positive]
        return dxdy

    def ddxddy(self, x, **kwargs):
        ddxddy = np.zeros_like(self.positive, dtype=float)
        ddxddy[self.positive] = np.broadcast_to((-3 / (4 * x**(5/2))), self.positive.shape)[self.positive]
        ddxddy[~self.positive] = np.broadcast_to((3 / (4 * x**(5/2))), self.positive.shape)[~self.positive]
        return ddxddy
