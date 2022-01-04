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
        self.get_asymptotes(x)

    def get_asymptotes(self,x):
        pass

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
        x_min = (1.0+self.factor)*self.low
        x_max = (1.0-self.factor)*self.upp
        return np.clip(x, x_min, x_max, out=x)

class MMA(MMAp):
    """A variant of the MMA intervening variables.
       As the exponent p decreases, the approximation becomes more conservative.
       Only p<=1 are allowed.

    Includes the following set of mixed intervening variables:
        y_i = (U_i - x_i) ** p     ,  if dg_j/dx_i >= 0
        y_i = (x_i - L_i) ** p     ,  if dg_j/dx_i < 0
    """

    def __init__(self, factor=0.01, low=-10.0, upp=10.0):
        """
        Initialise the exponential intervening variable with a power.
        :param p: The power
        :param xlim: Minimum x, in case of negative p, to prevent division by 0
        """
        super().__init__(p=-1, factor=factor, low=low, upp=upp)

class MMA87A(MMA):
    def __init__(self, t=1/3, factor=0.01):
        super().__init__(factor=factor)
        self.low, self.upp = None, None
        self.t = t # t strictly between 0 and 1

    def get_asymptotes(self,x):
        self.low = x * self.t
        self.upp = x / self.t

class MMA87B(MMA):
    def __init__(self, s=0.7, l0=0, u0=5, lmin=-50, lmax=0.4, umin=2.5, umax=50, oscillation_tol = 1e-10, factor=0.01):
        super().__init__(factor=factor)
        self.s = s
        self.l0 = l0
        self.u0 = u0
        self.lmin = lmin
        self.lmax = lmax
        self.umin = umin
        self.umax = umax

    def get_asymptotes(self,x):
        self.xold2, self.xold1, self.x = self.xold1, self.x, x.copy()

        if self.xold2 is None:
            # Initial values of asymptotes
            self.low = self.l0*x
            self.upp = self.u0*x
        else:
            s = self.s * np.ones_like(x)
            oscillation = ((self.x - self.xold1) * (self.xold1 - self.xold2)) / self.dx

            # oscillating variables x_i are increase or decrease the factor
            s[oscillation > +self.oscillation_tol] = 1/s
            s[oscillation < -self.oscillation_tol] = s

            # update lower and upper asymptotes
            self.low = self.x - s*(x - self.low)
            self.upp = self.x + s*(self.upp - x)

        self.low = np.clip(self.low, self.lmin, self.lmax, out=self.low)
        self.upp = np.clip(self.upp, self.umin, self.umax, out=self.upp)






#
#
# class Svanberg2002(AsymptoteUpdateStrategy):
#     def __init__(self, x_min=0.0, x_max=1.0, asyinit=0.5, asyincr=1.2, asydecr=0.7, asybound=10.0, oscillation_tol=1e-10):
#         self.xold1, self.xold2 = None, None
#         self.x_min, self.x_max = x_min, x_max
#         self.x = None
#
#         self.asybound = asybound
#         self.asyinit = asyinit
#         self.asyincr = asyincr
#         self.asydecr = asydecr
#         self.oscillation_tol = oscillation_tol
#
#         # MMA parameter initialization
#         self.factor = None
#         self.dx = self.x_max - self.x_min
#
#         self.min_factor, self.max_factor = 1 / (self.asybound ** 2), self.asybound
#
#     def get_asymptotes(self, x):
#         self.xold2, self.xold1, self.x = self.xold1, self.x, x.copy()
#         """Increases or decreases the asymptotes interval based on oscillations in the design vector"""
#         if self.factor is None:
#             self.factor = np.full_like(self.x, self.asyinit)
#
#         if self.xold2 is None:
#             # Initial values of asymptotes
#             low = x - self.factor * self.dx
#             upp = x + self.factor * self.dx
#         else:
#             # Update scheme for asymptotes
#             # depending on if the signs of (x_k-xold) and (xold-xold2) are opposite, indicating an oscillation in xi
#             # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed
#
#             # check for oscillations in variables (if > 0: no oscillations, if < 0: oscillations)
#             oscillation = ((x - self.xold1) * (self.xold1 - self.xold2)) / self.dx
#
#             # oscillating variables x_i are increase or decrease the factor
#             self.factor[oscillation > +self.oscillation_tol] *= self.asyincr
#             self.factor[oscillation < -self.oscillation_tol] *= self.asydecr
#
#             # Clip the asymptote factor
#             np.clip(self.factor, self.min_factor, self.max_factor)
#
#             # update lower and upper asymptotes
#             low = x - self.factor * self.dx
#             upp = x + self.factor * self.dx
#
#         return low, upp
#
# class Svanberg1987_t(AsymptoteUpdateStrategy):
#     def __init__(self, t=1/4):
#         self.t = t
#
#     def get_asymptotes(self, x):
#         low = x*self.t
#         upp = x/self.t
#         return low, upp
#
# class Svanberg1987_s_fixed(AsymptoteUpdateStrategy):
#     def __init__(self, x_min=0.0, x_max=1.0, factor=0.1 ):
#         self.factor = factor
#         self.dx = x_max - x_min
#
#     def get_asymptotes(self, x):
#         low = x - self.factor * self.dx
#         upp = x + self.factor * self.dx
#         return low, upp
#
# class Svanberg1987_s_move(AsymptoteUpdateStrategy):
#     def __init__(self, x_min=0.0, x_max=1.0, factor=0.7):
#         self.xold1, self.xold2 = None, None
#         self.lowold1, self.uppold1 = None, None
#         self.dx = x_max - x_min
#         self.x = None
#         self.factor = factor
#
#     def get_asymptotes(self, x):
#         self.xold2, self.xold1, self.x = self.xold1, self.x, x.copy()
#
#
#         low = x - self.dx
#         upp = x + self.dx
#
#         if self.xold2 is not None:
#             # Update scheme for asymptotes
#             # depending on if the signs of (x_k-xold) and (xold-xold2) are opposite, indicating an oscillation in xi
#             # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed
#
#             # check for oscillations in variables (if > 0: no oscillations, if < 0: oscillations)
#             oscillation = ((x - self.xold1) * (self.xold1 - self.xold2)) / self.dx
#             factor = self.factor * np.ones_like(x)
#             # oscillating variables x_i are increase or decrease the factor
#             factor[oscillation < 0] = 1/self.factor
#
#             # update lower and upper asymptotes
#             low = x - self.factor * (self.xold1 - self.lowold1)
#             upp = x + self.factor * (self.uppold1 - self.xold1)
#
#         self.lowold1 = low
#         self.uppold1 = upp
#
#         return low, upp
#
#
#
#
#     @staticmethod
#     def clipSvanberg1987eq8(x, low, upp, factor):
#         x_min = low + factor * (x - low)
#         x_max = upp - factor * (upp - x)
#         return x_min, x_max
#
#     @staticmethod
#     def clipSvanberg1987eq24(low, upp, factor):
#         x_min = (1.0 + factor)*low
#         x_max = (1.0 - factor)*upp
#         return x_min, x_max