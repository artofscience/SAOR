## IMPORTS
import numpy as np


class QuadPoly1D:
    """
    This class is a single variable polynomial function that is used as a problem to validate multi-point Taylor-like
    expansions. It has a convex and a concave part, as well as 2 local minima.
    """
    def __init__(self):
        self.n = 1
        self.m = 1
        self.xmin = np.array([1.5])
        self.xmax = np.array([12.])
        self.x0  = np.array([11.])      # use 6.9 to avoid local minimum or 11.5 to see if it gets stuck

    def g(self, x):
        g_j = np.empty(self.m + 1)
        g_j[0] = -42.9214 + 72.2423 * x[0] - 41.9587 * x[0] ** 2 + 10.6299 * x[0] ** 3 - 1.2079 * x[0] ** 4 + 0.0508 * x[0] ** 5
        g_j[1] = -10 - (x[0] - 5) ** 3
        return g_j

    def dg(self, x):
        dg_j = np.empty((self.m + 1, self.n))
        dg_j[0, 0] = 72.2423 - 2 * 41.9587 * x[0] + 3 * 10.6299 * x[0] ** 2 - 4 * 1.2079 * x[0] ** 3 + 5 * 0.0508 * x[0] ** 4
        dg_j[1, 0] = -3 * (x[0] - 5) ** 2
        return dg_j

    def ddg(self, x):
        ddg_j = np.empty((self.m + 1, self.n))
        ddg_j[0, 0] = 1.016 * x[0]**3 - 14.4948 * x[0]**2 + 63.7794 * x[0] - 83.9174
        ddg_j[1, 0] = -6.0 * (x[0] - 5.0)
