import math
from abc import ABC, abstractmethod
import numpy as np


class MoveLimitStrategy(ABC):
    """
    This is an abstract implementation of a move limit strategy.
    """
    def __init__(self, xmin=-math.inf, xmax=math.inf, **kwargs):
        """
        This is the constructor of the abstract move limit strategy class.

        :param xmin: A problem's lower bound constraints with xmin.shape = [n]
        :param xmax: A problem's upper bound constraints with xmax.shape = [n]
        :param kwargs:
        """

        self.xmin = xmin
        self.xmax = xmax

    @abstractmethod
    def update(self, x, **kwargs):
        ...


class MoveLimitIntervening(MoveLimitStrategy):
    """
    This is a move limit strategy for the case of Taylor-like approximations wrt intervening variables.
    These intervening variables impose move limits on the allowable change of the design variables at each iteration.
    """

    def __init__(self, xmin=-math.inf, xmax=math.inf, move_limit=0.1, **kwargs):
        super().__init__(xmin, xmax, **kwargs)
        self.dx = xmax - xmin
        self.alpha = xmin + 0
        self.beta = xmax + 0
        self.move_limit = move_limit

    def update(self, x, **kwargs):
        """
        This method updates the allowable move limits from the current point.

        :param x: Design variable vector of size [n]
        :param kwargs: the inter.get_move_limit() is passed as a keyword argument in order to compute the maximum
                       variable change allowed by the intervening variables.
        :return: self.alpha, self.beta: lower and upper move-limits respectively
        """

        # limit change with the move limit
        zzl1 = x - self.move_limit * self.dx
        zzu1 = x + self.move_limit * self.dx

        # TODO: Alter this by implementing a `get_move_limit()` method in abstract class Intervening in order to avoid
        #  the `if` statement. Didn't do it already cause you don't have access to the variable size `n` there.
        # if intervening vars provide a type of 'move limit' (e.g. MMA), limit change in x_i wrt intermediate variables
        if (inter := kwargs.get('intervening')) and hasattr(inter, 'get_move_limit'):
            zzl2, zzu2 = inter.get_move_limit()
            self.alpha = np.maximum.reduce([zzl1, zzl2, self.xmin])  # gets the max for each row of (zzl1, zzl2, xmin)
            self.beta = np.minimum.reduce([zzu1, zzu2, self.xmax])   # gets the min for each row of (zzu1, zzu2, xmax)
        else:
            self.alpha = np.maximum.reduce([zzl1, self.xmin])
            self.beta = np.minimum.reduce([zzu1, self.xmax])

        return self.alpha, self.beta


class MoveLimitMMA(MoveLimitStrategy):
    """
    This is an adaptive move limit strategy extracted from the MMA algorithm.
    Oscillatory behaviour (wrt each variable) is detected as the optimization runs,
    and the allowable step-size for that variable is adjusted accordingly.
    This move-limit strategy can be applied to any intervening variable selected.
    """

    def __init__(self, xmin=-math.inf, xmax=math.inf, move_limit=0.1, **kwargs):
        super().__init__(xmin, xmax, **kwargs)
        self.dx = xmax - xmin
        self.alpha = xmin + 0
        self.beta = xmax + 0
        self.move_limit = move_limit
        self.x, self.xold1, self.xold2 = None, None, None
        self.low, self.upp = None, None
        self.ml_init = kwargs.get('ml_init', 0.5)
        self.ml_incr = kwargs.get('ml_incr', 1.2)
        self.ml_decr = kwargs.get('ml_decr', 0.7)
        self.ml_bound = kwargs.get('ml_bound', 10.0)
        self.ml_albefa = kwargs.get('ml_albefa', 0.1)
        self.factor = self.ml_init * np.ones(len(xmin))

    def update(self, x, **kwargs):
        """
        This method updates the allowable move limits from the current point.
        It has a similar structure to the asymptote update rule given by:
        http://www.ingveh.ulg.ac.be/uploads/education/meca-0027-1/MMA_DCAMM_1998.pdf

        :param x: Design variable vector of size [n]
        :param kwargs:
        :return: self.alpha, self.beta: lower and upper move limits respectively
        """

        self.xold2 = self.xold1
        self.xold1 = self.x
        self.x = x

        # limit change with the move limit
        zzl1 = self.x - self.move_limit * self.dx
        zzu1 = self.x + self.move_limit * self.dx

        # if intervening vars provide a type of 'move limit' (e.g. MMA), limit change in x_i wrt intermediate variables
        if self.xold2 is None:
            self.low = self.x - self.factor * self.dx
            self.upp = self.x + self.factor * self.dx
            self.alpha = np.maximum.reduce([zzl1, self.xmin])
            self.beta = np.minimum.reduce([zzu1, self.xmax])
        else:
            # check for oscillations in variables (if zzz > 0: no oscillations, if zzz < 0: oscillations)
            zzz = (self.x - self.xold1) * (self.xold1 - self.xold2)

            # oscillating variables x_i are assigned a factor of asydecr and non-oscillating to asyincr
            self.factor[zzz > 0] = self.ml_incr
            self.factor[zzz < 0] = self.ml_decr

            # update lower and upper 'asymptotes'
            self.low = self.x - self.factor * (self.xold1 - self.low)
            self.upp = self.x + self.factor * (self.upp - self.xold1)

            # check min and max bounds of asymptotes, as they cannot be too close or far from the variable (redundant?)
            lowmin = self.x - self.ml_bound * self.dx
            lowmax = self.x - 1 / (self.ml_bound ** 2) * self.dx
            uppmin = self.x + 1 / (self.ml_bound ** 2) * self.dx
            uppmax = self.x + self.ml_bound * self.dx

            # if given asymptotes cross boundaries put them to their max/min values (redundant?)
            self.low = np.clip(self.low, lowmin, lowmax)
            self.upp = np.clip(self.upp, uppmin, uppmax)

            # move limits from 'asymptotes'
            zzl2 = self.low + self.ml_albefa * (self.x - self.low)
            zzu2 = self.upp - self.ml_albefa * (self.upp - self.x)

            # final move limits (most conservative)
            self.alpha = np.maximum.reduce([zzl1, zzl2, self.xmin])  # gets the max for each row of (zzl1, zzl2, xmin)
            self.beta = np.minimum.reduce([zzu1, zzu2, self.xmax])   # gets the min for each row of (zzu1, zzu2, xmax)

        return self.alpha, self.beta
