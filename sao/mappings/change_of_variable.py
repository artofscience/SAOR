from abc import abstractmethod, ABC
from .mapping import Mapping, Linear, PositiveNegative
import numpy as np


class Exponential(Mapping):
    def __init__(self, p=1, mapping=Linear(), xlim=1e-10):
        super().__init__(mapping)
        assert p != 0, f"Invalid power x^{p}, will result in zero division."
        self.p, self.xlim = p, xlim

    def _clip(self, x): return np.maximum(x, self.xlim, out=x) if self.p < 0 else x

    def _g(self, x): return x ** self.p

    def _dg(self, x): return self.p * x ** (self.p - 1)

    def _ddg(self, x): return self.p * (self.p - 1) * x ** (self.p - 2)


class MMAp(PositiveNegative, ABC):
    def __init__(self, p=-1, factor=1e-3, low=-10.0, upp=10.0):
        super().__init__(Exponential(p), Exponential(p))
        self.low, self.upp = low, upp
        self.factor = factor

    def update(self, x0, dg0, ddg0=0):
        super().update(x0, dg0, ddg0)
        [self.low, self.upp] = self.get_asymptotes(x0)

    @abstractmethod
    def get_asymptotes(self, x): pass

    def _g(self, x):
        return super().g(np.where(self.positive, self.upp - x, x - self.low))

    def _dg(self, x):
        return super().dg(np.where(self.positive, self.upp - x, x - self.low)) * np.where(self.positive, -1, +1)

    def _ddg(self, x):
        return super().ddg(np.where(self.positive, self.upp - x, x - self.low))

    def clip(self, x):
        return np.clip(x, self.low + self.factor, self.upp - self.factor, out=x)


class ConLin(PositiveNegative):
    def __init__(self): super().__init__(Exponential(-1), Exponential(1))


class MMA(MMAp, ABC):
    def __init__(self): super().__init__(-1)
