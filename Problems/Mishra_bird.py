## IMPORTS
import numpy as np


## CLASS: Mishra's bird + constraint by https://en.wikipedia.org/wiki/Test_functions_for_optimization
class MishraBird:

    def __init__(self):
        self.n = 2
        self.m = 1
        self.xmin = np.array([0., 0.])
        self.xmax = np.array([10., 6.5])
        self.name = 'MishraBird'

    def response(self, x_k):
        g_j = np.empty(self.m + 1)

        g_j[0] = np.sin(x_k[1] - 6.5) * np.exp((1 - np.cos(x_k[0] - 10.)) ** 2) + \
                 np.cos(x_k[0] - 10.)*np.exp((1 - np.sin(x_k[1] - 6.5)) ** 2) + (x_k[0] - x_k[1] - 3.5) ** 2

        g_j[1] = (x_k[0] - 5.) ** 2 + (x_k[1] - 1.5) ** 2 - 25.

        return g_j

    def sensitivity(self, x_k):
        dg_j = np.empty((self.m + 1, self.n))

        dg_j[0, 0] = 2.0 * x_k[0] - 2.0 * x_k[1] + 2 * (1 - np.cos(x_k[0] - 10.0)) * \
                     np.exp((1 - np.cos(x_k[0] - 10.0)) ** 2)*np.sin(x_k[0] - 10.0) * np.sin(x_k[1] - 6.5) - \
                     np.exp((1 - np.sin(x_k[1] - 6.5)) ** 2) * np.sin(x_k[0] - 10.0) - 7.0

        dg_j[0, 1] = -2.0 * x_k[0] + 2.0 * x_k[1] - 2 * (1 - np.sin(x_k[1] - 6.5)) * \
                     np.exp((1 - np.sin(x_k[1] - 6.5)) ** 2) * np.cos(x_k[0] - 10.0) * np.cos(x_k[1] - 6.5) + \
                     np.exp((1 - np.cos(x_k[0] - 10.0))**2) * np.cos(x_k[1] - 6.5) + 7.0

        dg_j[1, 0] = 2.0 * x_k[0] - 10.0

        dg_j[1, 1] = 2.0 * x_k[1] - 3.0

        return dg_j
