## IMPORTS
import numpy as np
import math


## CLASS: see ReferenceFiles/Problems.pdf by https://en.wikipedia.org/wiki/Test_functions_for_optimization
class Townsend:

    def __init__(self):
        self.n = 2
        self.m = 1
        self.xmin = np.array([0., 0.])
        self.xmax = np.array([4.75, 4.25])
        self.name = 'Townsend'

    def response(self, x_k):
        g_j = np.empty(self.m + 1)
        t = math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)

        g_j[0] = - (np.cos((x_k[0] - 2.35) * (x_k[1] - 2.5))) ** 2 - (x_k[0] - 2.25) * \
                    np.sin(3 * (x_k[0] - 2.25) + x_k[1] - 2.5)

        g_j[1] = (x_k[0] - 2.25) ** 2 + (x_k[1] - 2.5) ** 2 - \
                 (2 * np.cos(t) - 0.5 * (np.cos(2 * t)) - 0.25 * np.cos(3 * t) - 1 / 8 * np.cos(4 * t)) ** 2 - \
                 (2 * np.sin(t)) ** 2

        return g_j

    def sensitivity(self, x_k):
        dg_j = np.empty((self.m + 1, self.n))

        dg_j[0, 0] = 3 * (2.25 - x_k[0]) * np.cos(3 * x_k[0] + x_k[1] - 9.25) + 2 * (x_k[1] - 2.5) * \
                     np.sin((x_k[0] - 2.35) * (x_k[1] - 2.5)) * np.cos((x_k[0] - 2.35) * (x_k[1] - 2.5)) - \
                     np.sin(3 * x_k[0] + x_k[1] - 9.25)

        dg_j[0, 1] = (2.25 - x_k[0]) * np.cos(3 * x_k[0] + x_k[1] - 9.25) + 2 * (x_k[0] - 2.35) * \
                     np.sin((x_k[0] - 2.35) * (x_k[1] - 2.5)) * np.cos((x_k[0] - 2.35) * (x_k[1] - 2.5))

        dg_j[1, 0] = 2.0 * x_k[0] - 3.24 * (0.72 - 0.32 * x_k[0]) * (0.444444444444444 * x_k[0] - 1) ** 2 / \
                     (0.81 * (0.444444444444444 * x_k[0] - 1) ** 2 + (0.4 * x_k[1] - 1) ** 2) ** 2 + \
                     (2.88 - 1.28 * x_k[0]) / (0.81 * (0.444444444444444 * x_k[0] - 1) ** 2 + (0.4 * x_k[1] - 1) ** 2) - \
                     0.64 * ((x_k[1] - 2.5) / np.sqrt(0.81 * (0.444444444444444 * x_k[0] - 1) ** 2 + (0.4 * x_k[1] - 1) ** 2) -
                             0.625 * np.cos(2 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) - 0.3125 *
                             np.cos(3 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) - 0.15625 *
                             np.cos(4 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5))) * \
                     (2 * (0.36 - 0.16 * x_k[0]) * (x_k[1] - 2.5) / (0.81 * (0.444444444444444 * x_k[0] - 1) ** 2 + (0.4 * x_k[1] - 1) ** 2) ** (3/2)
                      + 2.5 * (x_k[1] - 2.5) * np.sin(2 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) /
                      (5.0625 * (0.444444444444444 * x_k[0] - 1) ** 2 + 6.25 * (0.4 * x_k[1] - 1) ** 2) +
                      1.875 * (x_k[1] - 2.5) * np.sin(3 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) /
                      (5.0625 * (0.444444444444444 * x_k[0] - 1) ** 2 + 6.25 * (0.4 * x_k[1] - 1) ** 2) +
                      1.25 * (x_k[1] - 2.5) * np.sin(4 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) /
                      (5.0625 * (0.444444444444444 * x_k[0] - 1) ** 2 + 6.25 * (0.4 * x_k[1] - 1) ** 2)) - 4.5

        dg_j[1, 1] = 2.0 * x_k[1] - 3.24 * (0.8 - 0.32 * x_k[1]) * (0.444444444444444 * x_k[0] - 1) ** 2 / \
                     (0.81 * (0.444444444444444 * x_k[0] - 1) ** 2 + (0.4 * x_k[1] - 1) ** 2) ** 2 - \
                     0.64 * ((x_k[1] - 2.5) / np.sqrt(0.81 * (0.444444444444444 * x_k[0] - 1) ** 2 + (0.4 * x_k[1] - 1) ** 2) -
                             0.625 * np.cos(2 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) - 0.3125 *
                             np.cos(3 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) - 0.15625 *
                             np.cos(4 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5))) * \
                     (2 * (0.4 - 0.16 * x_k[1]) * (x_k[1] - 2.5) / (0.81 * (0.444444444444444 * x_k[0] - 1) ** 2 + (0.4 * x_k[1] - 1) ** 2) ** (3/2) +
                      2.5 * (2.25 - x_k[0]) * np.sin(2 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) /
                      (5.0625 * (0.444444444444444 * x_k[0] - 1) ** 2 + 6.25 * (0.4 * x_k[1] - 1) ** 2) +
                      1.875 * (2.25 - x_k[0]) * np.sin(3 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) /
                      (5.0625 * (0.444444444444444 * x_k[0] - 1) ** 2 + 6.25 * (0.4 * x_k[1] - 1) ** 2) + 1.25 *
                      (2.25 - x_k[0]) * np.sin(4 * math.atan2(x_k[0] - 2.25, x_k[1] - 2.5)) /
                      (5.0625 * (0.444444444444444 * x_k[0] - 1) ** 2 + 6.25 * (0.4 * x_k[1] - 1) ** 2) + 2 /
                      np.sqrt(0.81 * (0.444444444444444 * x_k[0] - 1) ** 2 + (0.4 * x_k[1] - 1) ** 2)) - 5.0

        return dg_j