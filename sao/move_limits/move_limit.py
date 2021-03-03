import numpy as np
import math


class MoveLimitStrategy:
    def __init__(self, xmin=-math.inf, xmax=math.inf, move_limit=0.2, **kwargs):
        self.xmin = xmin
        self.xmax = xmax
        self.dx = xmax - xmin
        self.alpha = xmin + 0
        self.beta = xmax + 0
        self.move_limit = move_limit

    def update_move_limit(self, x, **kwargs):

        # limit change with the move limit
        zzl1 = x - self.move_limit * self.dx
        zzu1 = x + self.move_limit * self.dx

        # if intervening vars provide a type of 'move limit' (e.g. MMA), limit change in x_i wrt intermediate variables
        if (inter := kwargs.get('intervening')) and hasattr(inter, 'get_bounds'):
            zzl2, zzu2 = inter.get_bounds()
            self.alpha = np.maximum.reduce([zzl1, zzl2, self.xmin])  # gets the max for each row of (zzl1, zzl2, xmin)
            self.beta = np.minimum.reduce([zzu1, zzu2, self.xmax])   # gets the min for each row of (zzu1, zzu2, xmax)
        else:
            self.alpha = np.maximum.reduce([zzl1, self.xmin])
            self.beta = np.minimum.reduce([zzu1, self.xmax])

        return self.alpha, self.beta
