from abc import ABC
import numpy as np


class Mapping(ABC):
    @property
    def name(self):
        return self.__class__.name

    def __init__(self, mapping=None):
        self.map = mapping if mapping is not None else Linear()

    def update(self, x0, dg0, ddg0=0):
        self.map.update(x0, dg0, ddg0)
        dg = self.map.dg(x0)
        self._update(self.map.g(x0), dg0 / dg, ddg0 / dg ** 2 - dg0 * self.map.ddg(x0) / dg ** 3)

    def clip(self, x): return self._clip(self.map.clip(x))

    def g(self, x): return self._g(self.map.g(x))

    def dg(self, x): return self._dg(self.map.g(x)) * self.map.dg(x)

    def ddg(self, x): return self._ddg(self.map.g(x)) * (self.map.dg(x)) ** 2 + \
                             self._dg(self.map.g(x)) * self.map.ddg(x)

    def _update(self, x0, dg0, ddg0=0): pass

    def _clip(self, x): return x

    def _g(self, x): return x

    def _dg(self, x): return np.ones_like(x)

    def _ddg(self, x): return np.zeros_like(x)


class Linear(Mapping):
    def __init__(self): pass

    def update(self, x0, dg0, ddg0=0): pass

    def g(self, x): return x

    def dg(self, x): return np.ones_like(x)

    def ddg(self, x): return np.zeros_like(x)


class Exponential(Mapping):
    def __init__(self, mapping=Linear(), p=1, xlim=1e-10):
        super().__init__(mapping)
        assert p != 0, f"Invalid power x^{p}, will result in zero division."
        self.p, self.xlim = p, xlim

    def _clip(self, x): return np.maximum(x, self.xlim, out=x) if self.p < 0 else x

    def _g(self, x): return x ** self.p

    def _dg(self, x): return self.p * x ** (self.p - 1)

    def _ddg(self, x): return self.p * (self.p - 1) * x ** (self.p - 2)


class PositiveNegative(Mapping):
    def __init__(self, left: Mapping, right: Mapping):
        self.left = left
        self.right = right
        self.positive = None

    def update(self, x0, dg0, ddg0=0):
        self.left.update(x0, dg0, ddg0)
        self.right.update(x0, dg0, ddg0)
        self.positive = dg0 >= 0

    def g(self, x): return np.where(self.positive, self.right.g(x), self.left.g(x))

    def dg(self, x): return np.where(self.positive, self.right.dg(x), self.left.dg(x))

    def ddg(self, x): return np.where(self.positive, self.right.ddg(x), self.left.ddg(x))

    def clip(self, x):
        self.left.clip(x)
        self.right.clip(x)
        return x


class ConLin(PositiveNegative):
    def __init__(self):
        super().__init__(Exponential(p=-1), Exponential(p=1))


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