## IMPORTS
from sao.problems.problem import Problem
import numpy as np


## CLASS: see reference_files/problems.pdf by https://en.wikipedia.org/wiki/Test_functions_for_optimization
class RosenCubic(Problem):

    def __init__(self):
        super().__init__()
        self.n = 2
        self.m = 2
        self.x_min = np.array([0., 0.])
        self.x_max = np.array([3., 3.])
        self.x0 = np.array([0.5, 0.3])
        self.name = 'RosenCubic'

    def g(self, x_k):
        g_j = np.empty(self.m + 1)

        g_j[0] = (2.5 - x_k[0]) ** 2 + 100. * ((x_k[1] - 0.5) - (x_k[0] - 1.5) ** 2) ** 2

        g_j[1] = (x_k[0] - 2.5) ** 3 - x_k[1] + 1.5

        g_j[2] = x_k[0] + x_k[1] - 4.

        return g_j

    def dg(self, x_k):
        dg_j = np.empty((self.m + 1, self.n))

        dg_j[0, 0] = 2.0 * x_k[0] + 506.25 * (2.66666666666667 - 1.77777777777778 * x_k[0]) * \
                     (0.444444444444444 * x_k[1] - (0.666666666666667 * x_k[0] - 1) ** 2 - 0.222222222222222) - 5.

        dg_j[0, 1] = 200.0 * x_k[1] - 450.0 * (0.666666666666667 * x_k[0] - 1) ** 2 - 100.0

        dg_j[1, 0] = 3 * (x_k[0] - 2.5) ** 2

        dg_j[1, 1] = -1

        dg_j[2, 0] = 1

        dg_j[2, 1] = 1

        return dg_j
