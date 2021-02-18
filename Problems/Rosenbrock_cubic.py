## IMPORTS
import numpy as np
from Problems.Problem import Problem


## CLASS: see ReferenceFiles/Problems.pdf by https://en.wikipedia.org/wiki/Test_functions_for_optimization
class RosenCubic(Problem):

    def __init__(self):
        super().__init__()
        self.n = 2
        self.m = 2
        self.xmin = np.array([0., 0.])
        self.xmax = np.array([3., 3.])
        self.x_init = np.array([0.5, 0.3])
        self.g = np.empty(self.m + 1, dtype=float)
        self.dg = np.empty((self.m + 1, self.n), dtype=float)
        self.ddg = np.empty((self.m + 1, self.n), dtype=float)
        self.name = 'RosenCubic'

    def get_g(self, x_k):
        self.g[0] = (2.5 - x_k[0]) ** 2 + 100. * ((x_k[1] - 0.5) - (x_k[0] - 1.5) ** 2) ** 2
        self.g[1] = (x_k[0] - 2.5) ** 3 - x_k[1] + 1.5
        self.g[2] = x_k[0] + x_k[1] - 4.

    def get_dg(self, x_k):
        self.dg[0, 0] = 2.0 * x_k[0] + 506.25 * (2.66666666666667 - 1.77777777777778 * x_k[0]) * \
                        (0.444444444444444 * x_k[1] - (0.666666666666667 * x_k[0] - 1) ** 2 - 0.222222222222222) - 5.
        self.dg[0, 1] = 200.0 * x_k[1] - 450.0 * (0.666666666666667 * x_k[0] - 1) ** 2 - 100.0
        self.dg[1, 0] = 3 * (x_k[0] - 2.5) ** 2
        self.dg[1, 1] = -1
        self.dg[2, 0] = 1
        self.dg[2, 1] = 1
