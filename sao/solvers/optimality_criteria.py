import numpy as np


class OptimalityCriteria:
    """
    Optimality criteria optimizer.
    :return:
    """

    def __init__(self):
        self.move = 0.2
        self.gt = 0
        self.tol = 1e-3

    def update(self, x, dg, lower=0, upper=1e9):
        x_new = np.zeros_like(x)
        g = 0
        while (upper - lower) / (lower + upper) > self.tol:
            middle = (lower + upper) / 2
            y = x * np.sqrt(-dg[0] / dg[1] / middle)
            x_new[:] = np.maximum(0.0, np.clip(np.minimum(1.0, np.minimum(x + self.move, y)), x - self.move, 1.0))
            g = self.gt + np.sum((dg[1] * (x_new - x)))
            if g > 0:
                lower = middle
            else:
                upper = middle
        self.gt = g
        return x_new
