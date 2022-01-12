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
        self.map = mapping
        self.g0, self.dgdy0 = None, None
        self.nresp, self.nvar = -1, -1

    def _update(self, x, f, df, ddf=None):
        self.nresp, self.nvar = df.shape
        assert len(x) == self.nvar, "Mismatch in number of design variables."
        assert len(f) == self.nresp, "Mismatch in number of responses."
        self.map.update(x, f, df, ddf)
        self.dgdy0 = df / self.map.dg(x)
        self.g0 = (f / self.nvar)[:, np.newaxis] - self.dgdy0 * self.map.g(x)

    def g(self, x):
        return self._g(x)

    def dg(self, x):
        return self._dg(x)

    def ddg(self, x):
        return self._ddg(x)

    def _g(self, x): return self.g0 + self.dgdy0 * self.map.g(x)

    def _dg(self, x): return self.dgdy0 * self.map.dg(x)

    def _ddg(self, x): return self.dgdy0 * self.map.ddg(x)
