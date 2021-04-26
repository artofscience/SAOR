import math
from abc import ABC, abstractmethod
import numpy as np


class MoveLimitStrategy(ABC):
    """
    This is an abstract implementation of a move limit strategy.
    """
    def __init__(self, xmin=-math.inf, xmax=math.inf, **kwargs):
        """
        Description.
        :param xmin:
        :param xmax:
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


class MoveLimit1(MoveLimitStrategy):
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
        self.x, self.xold1, self.xold2 = None, None, None
        self.ml_init = kwargs.get('ml_init', 0.5)
        self.ml_incr = kwargs.get('ml_incr', 1.2)
        self.ml_decr = kwargs.get('ml_decr', 0.7)
        self.ml_bound = kwargs.get('ml_bound', 10.0)
        self.ml_albefa = kwargs.get('ml_albefa', 0.1)
        self.factor = self.ml_init * np.ones(len(xmin))

    def update(self, x, **kwargs):
        self.xold2 = self.xold1
        self.xold1 = self.x
        self.x = x

        # limit change with the move limit
        zzl1 = x - self.move_limit * self.dx
        zzu1 = x + self.move_limit * self.dx

        # if intervening vars provide a type of 'move limit' (e.g. MMA), limit change in x_i wrt intermediate variables
        if self.xold2 is not None:
            zzz = (self.x - self.xold1) * (self.xold1 - self.xold2)
            self.factor[zzz > 0] = self.ml_incr
            self.factor[zzz < 0] = self.ml_decr

            # update lower and upper bounds
            lower = self.x - self.factor * (self.xold1 - self.alpha)
            upper = self.x + self.factor * (self.beta - self.xold1)

            zzl2 = lower + self.ml_albefa * (self.x - lower)
            zzu2 = upper - self.ml_albefa * (upper - self.x)

            # check max bounds
            zzl2_min = self.x - self.ml_bound * self.dx
            zzl2_max = self.x - 1 / (self.ml_bound ** 2) * self.dx
            zzu2_min = self.x + 1 / (self.ml_bound ** 2) * self.dx
            zzu2_max = self.x + self.ml_bound * self.dx

            # clip bounds
            zzl2 = np.clip(zzl2, zzl2_min, zzl2_max)
            zzu2 = np.clip(zzu2, zzu2_min, zzu2_max)

            self.alpha = np.maximum.reduce([zzl1, zzl2, self.xmin])  # gets the max for each row of (zzl1, zzl2, xmin)
            self.beta = np.minimum.reduce([zzu1, zzu2, self.xmax])   # gets the min for each row of (zzu1, zzu2, xmax)
        else:
            self.alpha = np.maximum.reduce([zzl1, self.xmin])
            self.beta = np.minimum.reduce([zzu1, self.xmax])

        return self.alpha, self.beta
