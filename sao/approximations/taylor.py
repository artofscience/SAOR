from .approximation import Approximation
import numpy as np


class Taylor1(Approximation):
    def g_approx(self, x):
        return self.g + self.dg.dot(x-self.x)

    def dg_approx(self, x):
        return self.dg

    def ddg_approx(self, x):
        return np.zeros_like(self.dg)


class Taylor2(Taylor1):
    def g_approx(self, x):
        return super().g_approx(x) + 0.5*self.ddg.dot((x-self.x)**2)

    def dg_approx(self, x):
        return super().dg_approx(x) + self.ddg.dot(x-self.x)

    def ddg_approx(self, x):
        return super().ddg_approx(x) + self.ddg
