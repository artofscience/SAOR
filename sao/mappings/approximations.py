import numpy as np
from .mapping import Mapping, Linear


class LinearApproximation(Mapping):
    def __init__(self, mapping=Linear()):
        super().__init__(mapping)
        self.g0, self.dg0 = None, None

    def _update(self, x0, dg0, ddg0=0):
        self.g0 = -dg0 * x0
        self.dg0 = dg0

    def _g(self, x): return self.g0 + self.dg0 * x

    def _dg(self, x): return self.dg0

    def _ddg(self, x): return np.zeros_like(x)


class DiagonalQuadraticApproximation(LinearApproximation):
    def __init__(self, mapping=Linear()):
        super().__init__(mapping)
        self.ddg0 = None

    def _update(self, x0, dg0, ddg0=0):
        super()._update(x0, dg0)
        self.g0 += 0.5 * ddg0 * x0 ** 2
        self.dg0 -= ddg0 * x0
        self.ddg0 = ddg0

    def _g(self, x): return self.g0 + self.dg0 * x + 0.5 * self.ddg0 * x ** 2

    def _dg(self, x): return self.dg0 + self.ddg0 * x

    def _ddg(self, x): return self.ddg0
