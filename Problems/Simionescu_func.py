## IMPORTS
import numpy as np
from Problems.AbstractProblem import Problem


## CLASS: see ReferenceFiles/Problems.pdf by https://en.wikipedia.org/wiki/Test_functions_for_optimization
class Simionescu(Problem):

    def __init__(self):
        super().__init__(self)
        self.n = 2
        self.m = 1
        self.xmin = np.array([0., 0.])
        self.xmax = np.array([2.5, 2.5])
        self.x_init = np.array([0.6, 0.5])
        self.g = np.empty(self.m + 1, dtype=float)
        self.dg = np.empty((self.m + 1, self.n), dtype=float)
        self.ddg = np.empty((self.m + 1, self.n), dtype=float)
        self.name = 'Simionescu'

    def get_g(self, x_k):
        self.g[0] = 0.1 * (x_k[0] - 1.25) * (x_k[1] - 1.25)
        self.g[1] = (x_k[0] - 1.25) ** 2 + (x_k[1] - 1.25) ** 2 - \
                    (1. + 0.2 * np.cos(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25)))) ** 2

    def get_dg(self, x_k):
        self.dg[0, 0] = 0.1*x_k[1] - 0.125
        self.dg[0, 1] = 0.1*x_k[0] - 0.125
        self.dg[1, 0] = 2.0 * x_k[0] - 2.5 + 3.2 * (0.2 * np.cos(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25))) + 1.0) * \
                        np.sin(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25))) / ((x_k[1] - 1.25) * (1.0 * (0.8 * x_k[0] - 1) ** 2 / (0.8 * x_k[1] - 1) ** 2 + 1))
        self.dg[1, 1] = 2.0 * x_k[1] - 2.048 * (x_k[0] - 1.25) * (0.2 * np.cos(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25))) + 1.0) * \
                        np.sin(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25))) / \
                        ((0.8 * x_k[1] - 1) ** 2 * (1.0 * (0.8 * x_k[0] - 1) ** 2 / (0.8 * x_k[1] - 1) ** 2 + 1)) - 2.5
