from abc import ABC
import numpy as np


class Mapping(ABC):
    def __init__(self, mapping=None):
        self.map = mapping if mapping is not None else Linear()

    @property
    def name(self):
        return self.__class__.name

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


class Linear(Mapping, ABC):
    def __init__(self): pass

    def update(self, x0, dg0, ddg0=0): pass

    def g(self, x): return x

    def dg(self, x): return np.ones_like(x)

    def ddg(self, x): return np.zeros_like(x)


class MixedMapping(Mapping):
    def __init__(self, n: int, m: int, default: Mapping = Linear()):
        self.map = [(default, np.arange(0, m), np.arange(0, n))]

    def __setitem__(self, key, inter: Mapping):
        self.map.append((inter, key[0], key[1]))

    def eval(self, x, fn: callable):
        out = np.ones((len(self.map[0][1]), x.shape[0]), dtype=float)
        for mp, responses, variables in self.map:
            out[np.ix_(responses, variables)] = fn(mp, x[variables])
        return out

    def g(self, x):
        return self.eval(x, lambda cls, y: cls.g(y))

    def dg(self, x):
        return self.eval(x, lambda cls, y: cls.dg(y))

    def ddg(self, x):
        return self.eval(x, lambda cls, y: cls.ddg(y))

    def update(self, x0, dg0, ddg0=0):
        for mp, resp, var in self.map:
            mp.update(x0[var], dg0[np.ix_(resp, var)], ddg0=0)
        return self

    def clip(self, x):
        for mp, _, var in self.map:
            mp.clip(x[var])
        return x


class TwoMap(Mapping, ABC):
    def __init__(self, left: Mapping, right: Mapping):
        self.left = left
        self.right = right

    def update(self, x0, dg0, ddg0=0):
        self.left.update(x0, dg0, ddg0)
        self.right.update(x0, dg0, ddg0)

    def clip(self, x):
        self.left.clip(x)
        self.right.clip(x)
        return x


class Sum(TwoMap):
    def g(self, x): return self.left.g(x) + self.right.g(x)

    def dg(self, x): return self.left.dg(x) + self.right.dg(x)

    def ddg(self, x): return self.left.ddg(x) + self.right.ddg(x)


class Conditional(TwoMap, ABC):
    def __init__(self, left: Mapping, right: Mapping):
        super().__init__(left, right)
        self.condition = None

    def update(self, x0, dg0, ddg0=0):
        super().update(x0, dg0, ddg0)
        self.condition = dg0 >= 0

    def g(self, x): return np.where(self.condition, self.right.g(x), self.left.g(x))

    def dg(self, x): return np.where(self.condition, self.right.dg(x), self.left.dg(x))

    def ddg(self, x): return np.where(self.condition, self.right.ddg(x), self.left.ddg(x))
