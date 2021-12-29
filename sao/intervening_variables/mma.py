import numpy as np
from .exponential import Exponential
from .split import PositiveNegative

from sao.intervening_variables.asymptote_update_strategies import Svanberg2002


class MMAp(PositiveNegative):

    """The MMA algorithm, given by: http://www.ingveh.ulg.ac.be/uploads/education/meca-0027-1/MMA_DCAMM_1998.pdf

    Includes the following set of mixed intervening variables:
        y_i = 1 / (U_i - x_i)     ,  if dg_j/dx_i >= 0
        y_i = 1 / (x_i - L_i)     ,  if dg_j/dx_i < 0

    where,
        U_i := Upper asymptote (acts as an upper move-limit & adjusts the approximation's convexity)
        L_i := Lower asymptote (acts as a lower move-limit & adjusts the approximation's convexity)
    """


    def __init__(self, p=-1, albefa=0.1, updaterule=Svanberg2002(x_min=0.0, x_max=1.0, asyinit=0.5, asyincr=1.2, asydecr=0.7,
                                                     asybound=10.0, oscillation_tol=1e-10)):
        super().__init__(Exponential(p), Exponential(p))

        self.low, self.upp = None, None
        self.update_rule = updaterule
        self.albefa = albefa


    def update(self, x, f, df, *args, **kwargs):
        """Update state of previous iterations."""

        self.positive = df >= 0         # size of [m_p, n_l]
        [self.low, self.upp] = self.update_rule.get_asymptotes(x)


    def y(self, x):
        return super().y(np.where(self.positive, self.upp - x, x - self.low))

    def dydx(self, x):
        g_x = np.where(self.positive, self.upp - x, x - self.low)
        dg_x = np.where(self.positive, -1, +1)
        return super().dydx(g_x) * dg_x

    def ddyddx(self, x):
        g_x = np.where(self.positive, self.upp - x, x - self.low)
        return super().ddyddx(g_x)

    # def get_move_limit(self,x):
    #     x_min = self.low + self.albefa * (x - self.low)
    #     x_max = self.upp - self.albefa * (self.upp - x)
    #     return x_min, x_max

    def clip(self, x):
        """
        Clips a vector x between the lower and upper asymptote, with minimum
        safety distance `albefa` to the asymptotes.
        :param x: The vector to be clipped
        :return: Clipped vector (reference of x)
        """
        [x_min, x_max] = clipSvanberg1987eq8(x, self.low, self.upp, self.albefa)

        return np.clip(x, x_min, x_max, out=x)

    @staticmethod
    def clipSvanberg1987eq8(x, low, upp, factor):
        x_min = low + factor * (x - low)
        x_max = upp - factor * (upp - x)
        return x_min, x_max

    @staticmethod
    def clipSvanberg1987eq24(low, upp, factor):
        x_min = (1.0 + factor)*low
        x_max = (1.0 - factor)*upp
        return x_min, x_max



class MMA(MMAp):
    """A variant of the MMA intervening variables.
       As the exponent p decreases, the approximation becomes more conservative.
       Only p<=1 are allowed.

    Includes the following set of mixed intervening variables:
        y_i = (U_i - x_i) ** p     ,  if dg_j/dx_i >= 0
        y_i = (x_i - L_i) ** p     ,  if dg_j/dx_i < 0
    """


    def __init__(self, xmin=0.0, xmax=1.0, asyinit=0.5, asyincr=1.2, asydecr=0.7, asybound=10.0, albefa=0.1,
                 oscillation_tol=1e-10):
        """
        Initialise the exponential intervening variable with a power.
        :param p: The power
        :param xlim: Minimum x, in case of negative p, to prevent division by 0
        """
        super().__init__(p=-1, albefa=albefa, updaterule=Svanberg2002(xmin=xmin, xmax=xmax, asyinit=asyinit, asyincr=asyincr, asydecr=asydecr,
                         asybound=asybound, oscillation_tol=oscillation_tol)                         )