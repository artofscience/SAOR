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
        self.ml_init = kwargs.get('asyinit', 0.5)
        self.ml_incr = kwargs.get('asyincr', 1.1)
        self.ml_decr = kwargs.get('asydecr', 0.7)
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
            zzl2 = self.x - self.factor * (self.xold1 - self.alpha)
            zzu2 = self.x + self.factor * (self.beta - self.xold1)

            self.alpha = np.maximum.reduce([zzl1, zzl2, self.xmin])  # gets the max for each row of (zzl1, zzl2, xmin)
            self.beta = np.minimum.reduce([zzu1, zzu2, self.xmax])  # gets the min for each row of (zzu1, zzu2, xmax)
        else:
            self.alpha = np.maximum.reduce([zzl1, self.xmin])
            self.beta = np.minimum.reduce([zzu1, self.xmax])

        return self.alpha, self.beta
