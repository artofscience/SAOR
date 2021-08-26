## IMPORTS
import numpy as np


class Polynomial2D:
    """
    This a 2D non-convex problem with 4 local minima that is used to simulate Topology Optimization problems.
    It is used throughout the paper to show the influence of different aspects of SAO.
    """
    def __init__(self):
        self.n = 2
        self.m = 3
        self.x_min = np.array([0.1, 0.1])
        self.x_max = np.array([3.3, 3.3])
        self.x0  = np.array([2, 2.1])

    def g(self, x):
        g_j = np.empty(self.m + 1)
        g_j[0] = x[0] ** 4 - 6.8 * x[0] ** 3 + 14.8 * x[0] ** 2 - 10.6 * x[0] + \
                 x[1] ** 4 - 7.0 * x[1] ** 3 + 16.0 * x[1] ** 2 - 13.0 * x[1] + 6.4
        g_j[1] = x[0] ** 3 - 5.1 * x[0] ** 2 + 8.67 * x[0] + x[1] ** 3 - 3.9 * x[1] ** 2 + 5.07 * x[1] - 9.11
        # g_j[2] = -1.728 * x[0] ** 3 + 7.344 * x[0] ** 2 - 10.404 * x[0] - \
        #          0.125 * x[1] ** 3 + 0.675 * x[1] ** 2 - 1.215 * x[1] + 4.842
        g_j[2] = -12.5 * x[0] - x[1] + 11
        g_j[3] = 0.125 * x[0] ** 3 - 0.675 * x[0] ** 2 + 1.215 * x[0] - \
                 x[1] ** 3 + 4.2 * x[1] ** 2 - 5.88 * x[1] + 1.015
        return g_j

    def dg(self, x):
        dg_j = np.empty((self.m + 1, self.n))
        dg_j[0, 0] = 4 * x[0] ** 3 - 20.4 * x[0] ** 2 + 29.6 * x[0] - 10.6
        dg_j[0, 1] = 4 * x[1] ** 3 - 21.0 * x[1] ** 2 + 32.0 * x[1] - 13.0
        dg_j[1, 0] = 3 * x[0]**2 - 10.2 * x[0] + 8.67
        dg_j[1, 1] = 3 * x[1] ** 2 - 7.8 * x[1] + 5.07
        # dg_j[2, 0] = -5.184 * x[0] ** 2 + 14.688 * x[0] - 10.404
        # dg_j[2, 1] = -0.375 * x[1] ** 2 + 1.35 * x[1] - 1.215
        dg_j[2, 0] = -12.5
        dg_j[2, 1] = -1.
        dg_j[3, 0] = 0.375 * x[0] ** 2 - 1.35 * x[0] + 1.215
        dg_j[3, 1] = -3 * x[1] ** 2 + 8.4 * x[1] - 5.88
        return dg_j

    def ddg(self, x):
        ddg_j = np.empty((self.m + 1, self.n))
        ddg_j[0, 0] = 12 * x[0] ** 2 - 40.8 * x[0] + 29.6
        ddg_j[0, 1] = 12 * x[1] ** 2 - 42.0 * x[1] + 32.0
        ddg_j[1, 0] = 6 * x[0] - 10.2
        ddg_j[1, 1] = 6 * x[1] - 7.8
        # ddg_j[2, 0] = -10.368 * x[0] + 14.688
        # ddg_j[2, 1] = -0.75 * x[1] + 1.35
        ddg_j[2, 0] = 0.
        ddg_j[2, 1] = 0.
        ddg_j[3, 0] = 0.75 * x[0] - 1.35
        ddg_j[3, 1] = -6 * x[1] + 8.4
        return ddg_j
