import numpy as np

from .exponential import Exponential
from .split import PositiveNegative


class MMAp(PositiveNegative):
    """The MMA algorithm, given by: http://www.ingveh.ulg.ac.be/uploads/education/meca-0027-1/MMA_DCAMM_1998.pdf

    Includes the following set of mixed intervening variables:
        y_i = 1 / (U_i - x_i)     ,  if dg_j/dx_i >= 0
        y_i = 1 / (x_i - L_i)     ,  if dg_j/dx_i < 0

    where,
        U_i := Upper asymptote (acts as an upper move-limit & adjusts the approximation's convexity)
        L_i := Lower asymptote (acts as a lower move-limit & adjusts the approximation's convexity)
    """

    def __init__(self, p=-1, factor=0.01, low=-10.0, upp=10.0):
        super().__init__(Exponential(p), Exponential(p))
        self.low, self.upp = None, None
        self.factor = factor

    def update(self, x, f, df, *args, **kwargs):
        """Update state of previous iterations."""
        self.positive = df >= 0
        [self.low, self.upp] = self.get_asymptotes(x)

    def get_asymptotes(self, x):
        return 1.0 * self.low, 1.0 * self.upp

    def y(self, x):
        return super().y(np.where(self.positive, self.upp - x, x - self.low))

    def dydx(self, x):
        g_x = np.where(self.positive, self.upp - x, x - self.low)
        dg_x = np.where(self.positive, -1, +1)
        return super().dydx(g_x) * dg_x

    def ddyddx(self, x):
        g_x = np.where(self.positive, self.upp - x, x - self.low)
        return super().ddyddx(g_x)

    def clip(self, x):
        """
        Clips a vector x between the lower and upper asymptote, with minimum
        safety distance `albefa` to the asymptotes.
        :param x: The vector to be clipped
        :return: Clipped vector (reference of x)
        """
        x_min = (1.0 + self.factor) * self.low
        x_max = (1.0 - self.factor) * self.upp
        return np.clip(x, x_min, x_max, out=x)


class MMA87A(MMAp):
    """
    The MMA as used in (Svanberg 1987) - "Test problem 1: cantilever beam"
    """

    def __init__(self, t=1 / 3, p=-1, factor=0.01):
        super().__init__(p=p, factor=factor)
        self.low, self.upp = None, None
        self.t = t  # t strictly between 0 and 1

    def get_asymptotes(self, x):
        low = x * self.t
        upp = x / self.t
        return low, upp


class MMA87B(MMAp):
    """
    The MMA as used in (Svanberg 1987) - "Test problem 2: 8-bar truss"
    """

    def __init__(self, s=0.7, l0=0, u0=5, lmin=-50, lmax=0.4, umin=2.5, umax=50, oscillation_tol=1e-10, p=-1,
                 factor=0.01):
        super().__init__(p=p, factor=factor)
        self.x, self.xold1, self.xold2 = None, None, None
        self.s = s
        self.l0 = l0
        self.u0 = u0
        self.lmin = lmin
        self.lmax = lmax
        self.umin = umin
        self.umax = umax
        self.oscillation_tol = oscillation_tol

    def get_asymptotes(self, x):
        self.xold2, self.xold1, self.x = self.xold1, self.x, x.copy()

        if self.xold2 is None:
            # Initial values of asymptotes
            low = self.l0 * x
            upp = self.u0 * x
        else:
            s = self.s * np.ones_like(x)
            oscillation = ((self.x - self.xold1) * (self.xold1 - self.xold2)) / self.dx

            # oscillating variables x_i are increase or decrease the factor
            s[oscillation > +self.oscillation_tol] = 1 / s
            s[oscillation < -self.oscillation_tol] = s

            # update lower and upper asymptotes
            low = self.x - s * (x - self.low)
            upp = self.x + s * (self.upp - x)

        low = np.clip(low, self.lmin, self.lmax, out=low)
        upp = np.clip(upp, self.umin, self.umax, out=upp)
        return low, upp


class MMA87C(MMAp):
    """
    The MMA as used in (Svanberg 1987) - "Test problem 3: 2-bar truss"
    """

    def __init__(self, sdecr=0.75, sincr=2, x_min=-1000, x_max=1000,
                 oscillation_tol=1e-10, p=-1, factor=0.01):
        super().__init__(p=p, factor=factor)
        self.x, self.xold1, self.xold2 = None, None, None
        self.lowold, self.uppold = None, None
        self.sdecr = sdecr
        self.sincr = sincr
        self.dx = x_max - x_min
        self.oscillation_tol = oscillation_tol

    def get_asymptotes(self, x):
        self.xold2, self.xold1, self.x = self.xold1, self.x, x.copy()

        if self.xold2 is None:
            # Initial values of asymptotes
            low = x - self.dx
            upp = x + self.dx
        else:
            s = np.ones_like(x)
            oscillation = ((self.x - self.xold1) * (self.xold1 - self.xold2)) / self.dx

            # oscillating variables x_i are increase or decrease the factor
            s[oscillation > +self.oscillation_tol] = self.sdecr
            s[oscillation < -self.oscillation_tol] = self.sincr

            # update lower and upper asymptotes
            low = self.x - s * (self.xold1 - self.lowold)
            upp = self.x + s * (self.uppold - self.xold1)

        self.lowold = low
        self.uppold = upp
        return low, upp


class MMA02(MMAp):
    def __init__(self, x_min=0.0, x_max=1.0, sinit=0.5, sincr=1.2, sdecr=0.7, asybound=10.0, oscillation_tol=1e-10,
                 p=-1, factor=0.01):
        super().__init__(p=p, factor=factor)
        self.x, self.xold1, self.xold2 = None, None, None
        self.dx = x_max - x_min

        self.asybound = asybound
        self.sinit = sinit
        self.sincr = sincr
        self.sdecr = sdecr
        self.oscillation_tol = oscillation_tol

        self.dist = None
        self.dist_min, self.dist_max = 1 / (self.asybound ** 2), self.asybound

    def get_asymptotes(self, x):
        self.xold2, self.xold1, self.x = self.xold1, self.x, x.copy()
        """Increases or decreases the asymptotes interval based on oscillations in the design vector"""
        if self.dist is None:
            self.dist = np.full_like(self.x, self.sinit)

        if self.xold2 is None:
            # Initial values of asymptotes
            low = x - self.dist * self.dx
            upp = x + self.dist * self.dx
        else:
            # Update scheme for asymptotes
            # depending on if the signs of (x_k-xold) and (xold-xold2) are opposite, indicating an oscillation in xi
            # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed

            # check for oscillations in variables (if > 0: no oscillations, if < 0: oscillations)
            oscillation = ((x - self.xold1) * (self.xold1 - self.xold2)) / self.dx

            # oscillating variables x_i are increase or decrease the factor
            self.dist[oscillation > +self.oscillation_tol] *= self.sincr
            self.dist[oscillation < -self.oscillation_tol] *= self.sdecr

            # Clip the asymptote factor
            np.clip(self.dist, self.dist_min, self.dist_max)

            # update lower and upper asymptotes
            low = x - self.dist * self.dx
            upp = x + self.dist * self.dx
        return low, upp
