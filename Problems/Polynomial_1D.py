## IMPORTS
import numpy as np


class QuadPoly1:
    """
    This class is a 1D problem with a polynomial objective function and a polynomial constraint that is used to validate
    multi-point Taylor-like expansions. Both functions have a convex and a concave part. Moreover, the objective has 2
    local minima.
    """
    def __init__(self):
        self.n = 1
        self.m = 1
        self.xmin = np.array([1.5])
        self.xmax = np.array([12.])
        self.x0  = np.array([5.9])      # use 6.9 to avoid local minimum or 11.5 to see if it gets stuck

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
        return ddg_j


class QuadPoly2:
    """
    This class is a 1D problem with a polynomial objective function and a polynomial constraint that is used to validate
    multi-point Taylor-like expansions. Both functions have a convex and a concave part. Moreover, the objective has 2
    local minima.
    """
    def __init__(self):
        self.n = 1
        self.m = 1
        self.xmin = np.array([-2.])
        self.xmax = np.array([1.5])
        self.x0  = np.array([1.2])      # use 6.9 to avoid local minimum or 11.5 to see if it gets stuck

    def g(self, x):
        g_j = np.empty(self.m + 1)
        g_j[0] = 4 - x[0] - 2*x[0]**2 + 1.2*x[0]**3 + x[0]**4
        g_j[1] = -1 - (x[0] - 0.5)**3
        return g_j

    def dg(self, x):
        dg_j = np.empty((self.m + 1, self.n))
        dg_j[0, 0] = -1 - 4*x[0] + 3.6*x[0]**2 + 4*x[0]**3
        dg_j[1, 0] = -3*(x[0] - 0.5)**2
        return dg_j

    def ddg(self, x):
        ddg_j = np.empty((self.m + 1, self.n))
        ddg_j[0, 0] = -4 + 7.2*x[0] + 12*x[0]**2
        ddg_j[1, 0] = -6*(x[0] - 0.5)
        return ddg_j
