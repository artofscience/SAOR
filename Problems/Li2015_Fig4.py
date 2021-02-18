## IMPORTS
import numpy as np
from Problems.Problem import Problem


## CLASS: This is the Fig. 4 of https://www.sciencedirect.com/science/article/abs/pii/S004579491500022X
class Li2015Fig4(Problem):

    def __init__(self):
        super().__init__()
        self.n = 2
        self.m = 2
        self.c1 = 1.0
        self.c2 = 0.124
        self.xmin = np.array([0.2, 0.1])
        self.xmax = np.array([4., 1.6])
        self.x0 = np.array([2.5, 1.])
        self.g = np.empty(self.m + 1, dtype=float)
        self.dg = np.empty((self.m + 1, self.n), dtype=float)
        self.ddg = np.empty((self.m + 1, self.n), dtype=float)
        self.name = 'Li2015Fig4'

    def get_g(self, x_k):
        self.g[0] = self.c1 * x_k[0] * np.sqrt(1 + x_k[1]**2)
        self.g[1] = self.c2 * np.sqrt(1 + x_k[1]**2) * (8/x_k[0] + 1/(x_k[0] * x_k[1])) - 1
        self.g[2] = self.c2 * np.sqrt(1 + x_k[1]**2) * (8/x_k[0] - 1/(x_k[0] * x_k[1])) - 1

    def get_dg(self, x_k):
        self.dg[0, 0] = self.c1 * np.sqrt(1 + x_k[1]**2)
        self.dg[0, 1] = self.c1 * x_k[0] * 1/(2*np.sqrt(1 + x_k[1]**2)) * 2 * x_k[1]
        self.dg[1, 0] = self.c2 * np.sqrt(1 + x_k[1]**2) * (-8/x_k[0]**2 - 1/(x_k[0]**2 * x_k[1]))
        self.dg[1, 1] = self.c2 * 8/x_k[0] * 1/(2 * np.sqrt(1 + x_k[1]**2)) * 2 * x_k[1] + \
                        self.c2/x_k[0] * (-1/x_k[1]**2 * np.sqrt(1 + x_k[1]**2) + 1/(np.sqrt(1 + x_k[1]**2)))
        self.dg[2, 0] = self.c2 * np.sqrt(1 + x_k[1]**2) * (-8/x_k[0]**2 + 1/(x_k[0]**2 * x_k[1]))
        self.dg[2, 1] = self.c2 * 8/x_k[0] * 1/(2 * np.sqrt(1 + x_k[1]**2)) * 2 * x_k[1] - \
                        self.c2/x_k[0] * (-1/x_k[1]**2 * np.sqrt(1 + x_k[1]**2) + 1/(np.sqrt(1 + x_k[1]**2)))

    def ddg(self, x_k):
        self.ddg[0, 0] = 0
        self.ddg[0, 1] = -self.c1 * x_k[0] * x_k[1]**2 / (x_k[1] ** 2 + 1.0)**(3/2) + self.c1 * x_k[0] / \
                         np.sqrt(x_k[1]**2 + 1.0)
        self.ddg[1, 0] = self.c2 * (16 / x_k[0]**3 + 2 / (x_k[0]**3 * x_k[1])) * np.sqrt(x_k[1]**2 + 1.0)
        self.ddg[1, 1] = -self.c2 * x_k[1]**2 * (8/x_k[0] + 1/(x_k[0] * x_k[1])) / (x_k[1]**2 + 1.0)**(3/2) + \
                         self.c2 * (8/x_k[0] + 1/(x_k[0]*x_k[1])) / np.sqrt(x_k[1]**2 + 1.0) - \
                         2 * self.c2 / (x_k[0] * x_k[1] * np.sqrt(x_k[1]**2 + 1.0)) + 2 * self.c2 * np.sqrt(x_k[1]**2 + 1.0) / \
                         (x_k[0] * x_k[1]**3)
        self.ddg[2, 0] = self.c2 * (16/x_k[0]**3 - 2/(x_k[0]**3 * x_k[1])) * np.sqrt(x_k[1]**2 + 1.0)
        self.ddg[2, 1] = - self.c2 * x_k[1]**2 * (8/x_k[0] - 1/(x_k[0] * x_k[1])) / (x_k[1]**2 + 1.0)**(3/2) + \
                         self.c2 * (8/x_k[0] - 1/(x_k[0] * x_k[1])) / np.sqrt(x_k[1]**2 + 1.0) + \
                         2 * self.c2 / (x_k[0] * x_k[1] * np.sqrt(x_k[1]**2 + 1.0)) - \
                         2 * self.c2 * np.sqrt(x_k[1]**2 + 1.0) / (x_k[0] * x_k[1]**3)
