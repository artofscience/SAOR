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

    def __init__(self, xmin=0.0, xmax=1.0, asyinit=0.5, asyincr=1.2, asydecr=0.7, asybound=10.0, albefa=0.1, oscillation_tol=1e-10):
        self.x = None
        self.xold1, self.xold2 = None, None
        self.low, self.upp = None, None

        self.xmin, self.xmax = xmin, xmax

        # MMA parameter initialization
        self.asyinit = asyinit
        self.asyincr = asyincr
        self.asydecr = asydecr
        self.asybound = asybound
        self.albefa = albefa
        self.osc_tol = oscillation_tol
        self.factor = None
        self.dx = xmax - xmin

        self.min_factor, self.max_factor = 1 / (self.asybound**2), self.asybound

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
            self.factor[oscillation > +self.osc_tol] *= self.asyincr
            self.factor[oscillation < -self.osc_tol] *= self.asydecr

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


class MMAp(MMA):
    """A variant of the MMA intervening variables.

    Includes the following set of mixed intervening variables:
        y_i = (U_i - x_i) ** p     ,  if dg_j/dx_i >= 0
        y_i = (x_i - L_i) ** p     ,  if dg_j/dx_i < 0
    """

    def __init__(self, p=-1, xmin=0.0, xmax=1.0, asyinit=0.5, asyincr=1.2, asydecr=0.7, asybound=10.0, albefa=0.1,
                 oscillation_tol=1e-10):
        """
        Initialise the exponential intervening variable with a power.
        :param p: The power
        :param xlim: Minimum x, in case of negative p, to prevent division by 0
        """
        assert p <= -1, f"Invalid power x^{p}, must be less than p<=-1."
        self.p = p
        super().__init__(xmin=xmin, xmax=xmax, asyinit=asyinit, asyincr=asyincr, asydecr=asydecr,
                         asybound=asybound, albefa=albefa, oscillation_tol=oscillation_tol)

    def y(self, x):
        y = np.zeros_like(self.positive, dtype=float)
        y[self.positive] = np.broadcast_to(((self.upp - x) ** self.p), self.positive.shape)[self.positive]
        y[~self.positive] = np.broadcast_to(((x - self.low) ** self.p), self.positive.shape)[~self.positive]
        return y

    def dydx(self, x):
        dydx = np.zeros_like(self.positive, dtype=float)
        dydx[self.positive] = np.broadcast_to((-self.p * (self.upp - x) ** (self.p-1)),
                                              self.positive.shape)[self.positive]
        dydx[~self.positive] = np.broadcast_to((self.p * (x - self.low) ** (self.p-1)),
                                               self.positive.shape)[~self.positive]
        return dydx

    def ddyddx(self, x):
        ddyddx = np.zeros_like(self.positive, dtype=float)
        ddyddx[self.positive] = np.broadcast_to((-self.p * (-(self.p-1)) * (self.upp-x) ** (self.p-2)),
                                                self.positive.shape)[self.positive]
        ddyddx[~self.positive] = np.broadcast_to((self.p * (self.p-1) * (x-self.low) ** (self.p-2)),
                                                 self.positive.shape)[~self.positive]
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
        dydx[~self.positive] = np.broadcast_to((-3 / (x - self.low)**4), self.positive.shape)[~self.positive]
        return dydx

    def ddyddx(self, x):
        ddyddx = np.zeros_like(self.positive, dtype=float)
        ddyddx[self.positive] = np.broadcast_to((12 / (self.upp-x)**5), self.positive.shape)[self.positive]
        ddyddx[~self.positive] = np.broadcast_to((12 / (x-self.low)**5), self.positive.shape)[~self.positive]
        return ddyddx
