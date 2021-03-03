from .approximation import Approximation
import numpy as np


class Taylor1(Approximation):
    def g(self, x):
        # return self.f + self.df.dot(x-self.x)
        return self.f + (self.df.dot(x-self.x) if len(x.shape) == 1 else np.diag(self.df.dot(x-self.x)))

    def dg(self, x):
        return self.df

    def ddg(self, x):
        return np.zeros(self.df.shape, dtype=float)


class Taylor2(Taylor1):
    def g(self, x):
        return super().g(x) + 0.5 * (self.ddf.dot((x-self.x)**2) if len(x.shape) == 1 else np.diag(self.ddf.dot(x-self.x)))

    def dg(self, x):
        return super().dg(x) + (self.ddf * (x-self.x).T)

    def ddg(self, x):
        return super().ddg(x) + self.ddf
