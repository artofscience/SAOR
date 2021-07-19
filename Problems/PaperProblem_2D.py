## IMPORTS
import numpy as np


class PaperProblem2D:
    """
    This a 2D non-convex problem with 4 local minima that is used to simulate Topology Optimization problems.
    It is used throughout the paper to show the influence of different aspects of SAO.
    """
    def __init__(self):
        self.n = 2
        self.m = 3
        self.xmin = np.array([-1.9, -1.9])
        self.xmax = np.array([1.3, 1.3])
        self.x0  = np.array([1.2, 1.2])

    def g(self, x):
        g_j = np.empty(self.m + 1)
        g_j[0] = 4 - x[0] - 2*x[0]**2 + 1.2*x[0]**3 + x[0]**4 - x[1] - 2*x[1]**2 + x[1]**3 + x[1]**4
        g_j[1] = -2 + (x[0] + 0.3)**3 + (x[1] + 0.7)**3
        g_j[2] = -1 - (1.2*x[0] + 0.7)**3 - (0.5*x[1] + 0.1)**3
        g_j[3] = -0.8 + (0.5*x[0] + 0.1)**3 - (x[1] + 0.6)**3
        return g_j

    def dg(self, x):
        dg_j = np.empty((self.m + 1, self.n))
        dg_j[0, 0] = -1 - 4*x[0] + 3.6*x[0]**2 + 4*x[0]**3
        dg_j[0, 1] = -1 - 4*x[1] + 3*x[1]**2 + 4*x[1]**3
        dg_j[1, 0] = 3 * (x[0] + 0.3) ** 2
        dg_j[1, 1] = 3 * (x[1] + 0.7) ** 2
        dg_j[2, 0] = -3.6 * (1.2*x[0] + 0.7) ** 2
        dg_j[2, 1] = -0.375 * (x[1] + 0.2) ** 2
        dg_j[3, 0] = 0.375 * (x[0] + 0.2) ** 2
        dg_j[3, 1] = -3 * (x[1] + 0.6) ** 2
        return dg_j

    def ddg(self, x):
        ddg_j = np.empty((self.m + 1, self.n))
        ddg_j[0, 0] = -4 + 7.2*x[0] + 12*x[0]**2
        ddg_j[0, 1] = -4 + 6*x[1] + 12*x[1]**2
        ddg_j[1, 0] = 6 * (x[0] + 0.3)
        ddg_j[1, 1] = 6 * (x[1] + 0.7)
        ddg_j[2, 0] = -10.368 * x[0] - 6.048
        ddg_j[2, 1] = -0.75 * x[1] - 0.15
        ddg_j[3, 0] = 0.75 * x[0] + 0.15
        ddg_j[3, 1] = -6 * x[1] - 3.6
        return ddg_j
