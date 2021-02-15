## IMPORTS
import numpy as np
from Problems.AbstractProblem import Problem


## CLASS: see ReferenceFiles/Problems.pdf by https://en.wikipedia.org/wiki/Test_functions_for_optimization
class MishraBird(Problem):

    def __init__(self):
        Problem.__init__(self)
        self.n = 2
        self.m = 1
        self.xmin = np.array([0., 0.])
        self.xmax = np.array([10., 6.5])
        self.x_init = np.array([7.3, 2.])
        self.g = np.empty(self.m + 1, dtype=float)
        self.dg = np.empty((self.m + 1, self.n), dtype=float)
        self.ddg = np.empty((self.m + 1, self.n), dtype=float)
        self.name = 'MishraBird'

    def get_g(self, x_k):
        self.g[0] = np.sin(x_k[1] - 6.5) * np.exp((1 - np.cos(x_k[0] - 10.)) ** 2) + \
              np.cos(x_k[0] - 10.)*np.exp((1 - np.sin(x_k[1] - 6.5)) ** 2) + (x_k[0] - x_k[1] - 3.5) ** 2
        self.g[1] = (x_k[0] - 5.) ** 2 + (x_k[1] - 1.5) ** 2 - 25.

    def get_dg(self, x_k):
        self.dg[0, 0] = 2.0 * x_k[0] - 2.0 * x_k[1] + 2 * (1 - np.cos(x_k[0] - 10.0)) * \
                   np.exp((1 - np.cos(x_k[0] - 10.0)) ** 2)*np.sin(x_k[0] - 10.0) * np.sin(x_k[1] - 6.5) - \
                   np.exp((1 - np.sin(x_k[1] - 6.5)) ** 2) * np.sin(x_k[0] - 10.0) - 7.0
        self.dg[0, 1] = -2.0 * x_k[0] + 2.0 * x_k[1] - 2 * (1 - np.sin(x_k[1] - 6.5)) * \
                   np.exp((1 - np.sin(x_k[1] - 6.5)) ** 2) * np.cos(x_k[0] - 10.0) * np.cos(x_k[1] - 6.5) + \
                   np.exp((1 - np.cos(x_k[0] - 10.0))**2) * np.cos(x_k[1] - 6.5) + 7.0
        self.dg[1, 0] = 2.0 * x_k[0] - 10.0
        self.dg[1, 1] = 2.0 * x_k[1] - 3.0
