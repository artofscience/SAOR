from .mapping import Mapping, Linear
import numpy as np


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
