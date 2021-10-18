from dataclasses import dataclass
import numpy as np
from .intervening import Intervening


@dataclass
class MMAOptions():
    """Options for the MMA-algorithms and their default values.

    Attributes:
        asyinit (float): Initial values for the asymptotes.
        asyincr (float): Factor to increase asymptotes.
        asydecr (float): Factor to decrease asymptotes.
        asybounds (float): The bounds for the asymptotes.
        albefa (float): Tolerance between the asymptote and bounds.
        oscillation_tol (float): Tolerance for oscillations detection between iterations.
    """
    asyinit: float = 0.5
    asyincr: float = 1.2
    asydecr: float = 0.7
    asybound: float = 0.5
    albefa: float = 0.1
    oscillation_tol: float = 1e-10


class MMA(Intervening):
    """The MMA algorithm, given by: http://www.ingveh.ulg.ac.be/uploads/education/meca-0027-1/MMA_DCAMM_1998.pdf

    Includes the following set of mixed intervening variables:
        y_i = 1 / (U_i - x_i)     ,  if dg_j/dx_i >= 0
        y_i = 1 / (x_i - L_i)     ,  if dg_j/dx_i < 0

    where,
        U_i := Upper asymptote (acts as an upper move-limit & adjusts the approximation's convexity)
        L_i := Lower asymptote (acts as a lower move-limit & adjusts the approximation's convexity)
    """

    def __init__(self, xmin=0.0, xmax=1.0, asyinit=0.5, asyincr=1.2, asydecr=0.7, asybound=0.5, albefa=0.1,
                 oscillation_tol=1e-10):
        self.x = None
        self.xold1, self.xold2 = None, None
        self.low, self.upp = None, None
        self.xmin, self.xmax = xmin, xmax

        self.asybound = asybound
        self.asyinit = asyinit
        self.asyincr = asyincr
        self.asydecr = asydecr
        self.albefa = albefa
        self.oscillation_tol = oscillation_tol

        # MMA parameter initialization
        self.factor = None
        self.dx = xmax - xmin

        self.min_factor, self.max_factor = 1 / (self.asybound ** 2), self.asybound

        # A boolean indicator array that keeps track of the positive (and negative) values of the variables
        self.positive = None

    def update(self, x, f, df, *args, **kwargs):
        """Update state of previous iterations."""
        self.xold2, self.xold1, self.x = self.xold1, self.x, x.copy()
        self.positive = df >= 0         # size of [m_p, n_l]
        self.get_asymptotes()

    def get_asymptotes(self):
        """Increases or decreases the asymptotes interval based on oscillations in the design vector"""
        if self.factor is None:
            self.factor = np.full_like(self.x, self.asyinit)

        if self.xold2 is None:
            # Initial values of asymptotes
            self.low = self.x - self.factor * self.dx
            self.upp = self.x + self.factor * self.dx
        else:
            # Update scheme for asymptotes
            # depending on if the signs of (x_k-xold) and (xold-xold2) are opposite, indicating an oscillation in xi
            # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed

            # check for oscillations in variables (if > 0: no oscillations, if < 0: oscillations)
            oscillation = ((self.x - self.xold1) * (self.xold1 - self.xold2)) / self.dx

            # oscillating variables x_i are increase or decrease the factor
            self.factor[oscillation > +self.oscillation_tol] *= self.asyincr
            self.factor[oscillation < -self.oscillation_tol] *= self.asydecr

            # Clip the asymptote factor
            np.clip(self.factor, self.min_factor, self.max_factor)

            # update lower and upper asymptotes
            self.low = self.x - self.factor * self.dx
            self.upp = self.x + self.factor * self.dx

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

    def get_move_limit(self):
        zzl2 = self.low + self.albefa * (self.x - self.low)
        zzu2 = self.upp - self.albefa * (self.upp - self.x)
        return zzl2, zzu2

    def clip(self, x):
        """
        Clips a vector x between the lower and upper asymptote, with minimum
        safety distance `albefa` to the asymptotes.
        :param x: The vector to be clipped
        :return: Clipped vector (reference of x)
        """
        dist = self.albefa * self.factor * self.dx
        l, u = self.low + dist, self.upp - dist
        return np.clip(x, l, u, out=x)


class MMAsquared(MMA):
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


class MMAcubed(MMA):
    """A variant of the MMA intervening variables.

    Includes the following set of mixed intervening variables:
        y_i = 1 / (U_i - x_i) ** 3     ,  if dg_j/dx_i >= 0
        y_i = 1 / (x_i - L_i) ** 3     ,  if dg_j/dx_i < 0
    """

    def y(self, x):
        y = np.zeros_like(self.positive, dtype=float)
        y[self.positive] = np.broadcast_to((1 / (self.upp - x)**3), self.positive.shape)[self.positive]
        y[~self.positive] = np.broadcast_to((1 / (x - self.low)**3), self.positive.shape)[~self.positive]
        return y

    def dydx(self, x):
        dydx = np.zeros_like(self.positive, dtype=float)
        dydx[self.positive] = np.broadcast_to((3 / (self.upp - x)**4), self.positive.shape)[self.positive]
        dydx[~self.positive] = np.broadcast_to((-3 / (x - self.low) ** 4), self.positive.shape)[~self.positive]
        return dydx

    def ddyddx(self, x):
        ddyddx = np.zeros_like(self.positive, dtype=float)
        ddyddx[self.positive] = np.broadcast_to((12 / (self.upp - x) ** 5), self.positive.shape)[self.positive]
        ddyddx[~self.positive] = np.broadcast_to((12 / (x - self.low) ** 5), self.positive.shape)[~self.positive]
        return ddyddx


class MMAfixedasy(MMA):

    def __init__(self, xmin=0.0, xmax=1.0, low=-0.1, upp=1.1):
        super().__init__(xmin, xmax)
        self.low0 = low
        self.upp0 = upp

    def get_asymptotes(self):
        self.low = self.low0
        self.upp = self.upp0

    def clip(self, x):
        return np.clip(x, self.low, self.upp, out=x)
