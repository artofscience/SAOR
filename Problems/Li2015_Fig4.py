## IMPORTS
import numpy as np


## CLASS: This is the Fig. 4 of https://www.sciencedirect.com/science/article/abs/pii/S004579491500022X
class Li2015Fig4:

    def __init__(self):
        self.n = 2
        self.m = 2
        self.c1 = 1.0
        self.c2 = 0.124
        self.xmin = np.array([0.2, 0.1])
        self.xmax = np.array([4., 1.6])
        self.name = 'Li2015Fig4'

    def response(self, x_k):
        g_j = np.empty(self.m + 1)
        g_j[0] = self.c1 * x_k[0] * np.sqrt(1 + x_k[1]**2)
        g_j[1] = self.c2 * np.sqrt(1 + x_k[1]**2) * (8/x_k[0] + 1/(x_k[0] * x_k[1])) - 1
        g_j[2] = self.c2 * np.sqrt(1 + x_k[1]**2) * (8/x_k[0] - 1/(x_k[0] * x_k[1])) - 1
        return g_j

    def sensitivity(self, x_k):
        x_k = x_k
        dg_j = np.empty((self.m + 1, self.n))
        dg_j[0, 0] = self.c1 * np.sqrt(1 + x_k[1]**2)
        dg_j[0, 1] = self.c1 * x_k[0] * 1/(2*np.sqrt(1 + x_k[1]**2)) * 2 * x_k[1]
        dg_j[1, 0] = self.c2 * np.sqrt(1 + x_k[1]**2) * (-8/x_k[0]**2 - 1/(x_k[0]**2 * x_k[1]))
        dg_j[1, 1] = self.c2 * 8/x_k[0] * 1/(2 * np.sqrt(1 + x_k[1]**2)) * 2 * x_k[1] + \
            self.c2/x_k[0] * (-1/x_k[1]**2 * np.sqrt(1 + x_k[1]**2) + 1/(np.sqrt(1 + x_k[1]**2)))
        dg_j[2, 0] = self.c2 * np.sqrt(1 + x_k[1]**2) * (-8/x_k[0]**2 + 1/(x_k[0]**2 * x_k[1]))
        dg_j[2, 1] = self.c2 * 8/x_k[0] * 1/(2 * np.sqrt(1 + x_k[1]**2)) * 2 * x_k[1] - \
                     self.c2/x_k[0] * (-1/x_k[1]**2 * np.sqrt(1 + x_k[1]**2) + 1/(np.sqrt(1 + x_k[1]**2)))
        return dg_j
