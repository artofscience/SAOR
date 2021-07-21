import numpy as np
from numpy import maximum as nax
from numpy import minimum as nin


class OptimalityCriteria:
    """
    Optimality criteria optimizer.
    :return:
    """

    def __init__(self):
        self.move = 0.2
        self.gt = 0
        self.tol = 1e-3

    def update(self, x, dg):
        lower = 0
        upper = 1e9
        x_new = np.zeros_like(x)
        g = 0

        while (upper - lower) / (lower + upper) > self.tol:
            middle = (lower + upper) / 2
            y = x * np.sqrt(-dg[0] / dg[1] / middle)
            x_new[:] = nax(0.0, nax(x - self.move, nin(1.0, nin(x + self.move, y))))
            g = self.gt + np.sum((dg[1] * (x_new - x)))
            if g > 0:
                lower = middle
            else:
                upper = middle
        self.gt = g
        return x_new
