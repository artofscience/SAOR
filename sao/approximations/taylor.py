from .approximation import Approximation
import numpy as np


class Taylor1(Approximation):
    def g_approx(self, x):
        # return self.g + self.dg.dot(x-self.x)
        return self.g + (self.dg.dot(x-self.x) if len(x.shape) == 1 else np.diag(self.dg.dot(x-self.x)))

    def dg_approx(self, x):
        return self.dg

    def ddg_approx(self, x):
        return np.zeros(self.dg.shape, dtype=float)


class Taylor2(Taylor1):
    def g_approx(self, x):
        return super().g_approx(x) + 0.5 * (self.ddg.dot((x-self.x)**2) if len(x.shape) == 1 else np.diag(self.ddg.dot(x-self.x)))

    def dg_approx(self, x):
        return super().dg_approx(x) + (self.ddg * (x-self.x).T)

    def ddg_approx(self, x):
        return super().ddg_approx(x) + self.ddg
