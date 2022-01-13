from abc import ABC
import numpy as np


class Mapping(ABC):
    @property
    def name(self):
        return self.__class__.name

    def __init__(self, mapping=None):
        self.map = mapping if mapping is not None else Linear()

    def update(self, x, f, df, ddf=None):
        self.map.update(x, f, df, ddf)
        self._update(x, f, df, ddf)

    def clip(self, x): return self._clip(self.map.clip(x))

    def g(self, x): return self._g(self.map.g(x))

    def dg(self, x): return self._dg(self.map.g(x)) * self.map.dg(x)

    def ddg(self, x): return self._ddg(self.map.g(x)) * (self.map.dg(x)) ** 2 + \
                             self._dg(self.map.g(x)) * self.map.ddg(x)

    def _update(self, x, f, df, ddf=None): pass

    def _clip(self, x): return x

    def _g(self, x): return x

    def _dg(self, x): return np.ones_like(x)

    def _ddg(self, x): return np.zeros_like(x)


class Linear(Mapping):
    def __init__(self): pass

    def update(self, x, f, df, ddf=None): pass

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


class Taylor1(Mapping):
    def __init__(self, mapping=Linear()):
        super().__init__(mapping)
        self.g0, self.dg0 = None, None
        self.c = 0.0

    def _update(self, x, f, df, ddf=None):
        self.c = f
        self.dg0 = df / self.map.dg(x)
        self.g0 = -self.dg0 * self.map.g(x)

    def _g(self, x): return self.g0 + self.dg0 * x

    def _dg(self, x): return self.dg0

    def _ddg(self, x): return np.zeros_like(x)


class Taylor2(Mapping):
    def __init__(self, mapping=Linear()):
        super().__init__(mapping)
        self.g0, self.dg0, self.ddg0 = None, None, None
        self.c = 0.0

    def _update(self, x, f, df, ddf=None):
        y0 = self.map.g(x)
        dy0 = self.map.dg(x)
        self.c = f
        self.ddg0 = ddf / dy0 ** 2 - df * self.map.ddg(x) / dy0 ** 3
        self.dg0 = df / dy0
        self.g0 = -self.dg0 * y0 + 0.5 * self.ddg0 * y0 ** 2
        self.tmp = self.dg0 - y0 * self.ddg0

    def _g(self, x): return self.g0 + self.tmp * x + 0.5 * self.ddg0 * x ** 2

    def _dg(self, x): return self.tmp + self.ddg0 * x

    def _ddg(self, x): return self.ddg0


class Taylor2A(Taylor1):
    def __init__(self, mapping=Linear()):
        super().__init__(mapping)
        self.ddg0 = None

    def _update(self, x, f, df, ddf=None):
        super()._update(x, f, df)
        y0 = self.map.g(x)
        dy0 = self.map.dg(x)
        self.ddg0 = ddf / dy0 ** 2 - df * self.map.ddg(x) / dy0 ** 3
        self.g0 += 0.5 * self.ddg0 * y0 ** 2
        self.tmp = self.dg0 - y0 * self.ddg0

    def _g(self, x): return self.g0 + self.tmp * x + 0.5 * self.ddg0 * x ** 2

    def _dg(self, x): return self.tmp + self.ddg0 * x

    def _ddg(self, x): return self.ddg0
