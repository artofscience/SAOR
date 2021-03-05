import numpy as np
import math
from .move_limit import MoveLimitStrategy


class MoveLimitIntervening(MoveLimitStrategy):
    """
    This is a move limit strategy for the case of Taylor-like approximations wrt intervening variables.
    These intervening variables impose move limits on the allowable change of the design variables at each iteration.
    """

    def __init__(self, xmin=-math.inf, xmax=math.inf, move_limit=0.1, **kwargs):
        super().__init__(xmin, xmax, move_limit, **kwargs)
        self.dx = xmax - xmin
        self.alpha = xmin + 0
        self.beta = xmax + 0

    def update(self, x, **kwargs):

        # limit change with the move limit
        zzl1 = x - self.move_limit * self.dx
        zzu1 = x + self.move_limit * self.dx

        # if intervening vars provide a type of 'move limit' (e.g. MMA), limit change in x_i wrt intermediate variables
        if (inter := kwargs.get('intervening')) and hasattr(inter, 'get_move_limit'):
            zzl2, zzu2 = inter.get_move_limit()
            self.alpha = np.maximum.reduce([zzl1, zzl2, self.xmin])  # gets the max for each row of (zzl1, zzl2, xmin)
            self.beta = np.minimum.reduce([zzu1, zzu2, self.xmax])   # gets the min for each row of (zzu1, zzu2, xmax)
        else:
            self.alpha = np.maximum.reduce([zzl1, self.xmin])
            self.beta = np.minimum.reduce([zzu1, self.xmax])

        return self.alpha, self.beta
