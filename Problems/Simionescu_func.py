## IMPORTS
import numpy as np


## CLASS: see ReferenceFiles/Problems.pdf by https://en.wikipedia.org/wiki/Test_functions_for_optimization
class Simionescu:

    def __init__(self):
        self.n = 2
        self.m = 1
        self.xmin = np.array([0., 0.])
        self.xmax = np.array([2.5, 2.5])
        self.name = 'Simionescu'

    def response(self, x_k):
        g_j = np.empty(self.m + 1)

        g_j[0] = 0.1 * (x_k[0] - 1.25) * (x_k[1] - 1.25)

        g_j[1] = (x_k[0] - 1.25) ** 2 + (x_k[1] - 1.25) ** 2 - \
                 (1. + 0.2 * np.cos(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25)))) ** 2

        return g_j

    def sensitivity(self, x_k):
        dg_j = np.empty((self.m + 1, self.n))

        dg_j[0, 0] = 0.1*x_k[1] - 0.125

        dg_j[0, 1] = 0.1*x_k[0] - 0.125

        dg_j[1, 0] = 2.0 * x_k[0] - 2.5 + 3.2 * (0.2 * np.cos(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25))) + 1.0) * \
                     np.sin(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25))) / ((x_k[1] - 1.25) * (1.0 * (0.8 * x_k[0] - 1) ** 2 / (0.8 * x_k[1] - 1) ** 2 + 1))

        dg_j[1, 1] = 2.0 * x_k[1] - 2.048 * (x_k[0] - 1.25) * (0.2 * np.cos(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25))) + 1.0) * \
                     np.sin(8 * np.arctan((x_k[0] - 1.25) / (x_k[1] - 1.25))) / \
                     ((0.8 * x_k[1] - 1) ** 2 * (1.0 * (0.8 * x_k[0] - 1) ** 2 / (0.8 * x_k[1] - 1) ** 2 + 1)) - 2.5

        return dg_j