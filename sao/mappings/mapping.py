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


class Linear(Mapping, ABC):
    def __init__(self): pass

    def update(self, x0, dg0, ddg0=0): pass

    def g(self, x): return x

    def dg(self, x): return np.ones_like(x)

    def ddg(self, x): return np.zeros_like(x)


def fill_set_when_emtpy(s, n):
    """Returns ``set(s)`` or a ``set(0..n)`` if ``set(s)`` is the empty set."""
    if s is None or s is ...:
        return set(range(n))
    try:
        s = set(s)
    except TypeError:
        s = {s}
    if len(s) == 0:
        return set(range(n))
    return s


class MixedMapping(Mapping):
    def __init__(self, nvar: int, nresp: int, default: Mapping = Linear()):
        self.default = default
        self.nvar = nvar
        self.nresp = nresp
        self.map = []
        responses = set(range(self.nresp))
        variables = set(range(self.nvar))
        self.__add_map(self.default, responses, variables)

    @property
    def maps(self):
        for mp, _, _ in self.map: yield mp

    def set_map(self, inter: Mapping, resp=set(), var=set()):
        new_resp = fill_set_when_emtpy(resp, self.nresp)
        new_vars = fill_set_when_emtpy(var, self.nvar)
        for _, responses, variables in self.map:
            for r in (new_resp & responses):
                diff = variables[r] - new_vars
                if len(diff) > 0:
                    responses.add(r)
                    variables[r] = diff
                else:
                    responses.remove(r)
                    del variables[r]
        return self.__add_map(inter, new_resp, new_vars)

    def __add_map(self, inter, resp=set(), var=set()):
        responses = fill_set_when_emtpy(resp, self.nresp)
        variables = fill_set_when_emtpy(var, self.nvar)
        self.map.append((inter, responses, {i: variables for i in responses}))

    def evaluate_for_each_response(self, x, fn: callable):
        out = np.zeros((self.nresp, x.shape[0]))
        for intv, responses, variables in self.map:
            y_all = fn(intv, x)
            for r in responses:
                var_indices = list(variables[r])
                if y_all.ndim > 1:
                    out[r, var_indices] += y_all[r, var_indices]
                else:
                    out[r, var_indices] += y_all[var_indices]
        return out

    def g(self, x):
        def g_of_x(cls, x):
            return cls.g(x)

        return self.evaluate_for_each_response(x, g_of_x)

    def dg(self, x):
        def dg_of_x(cls, x):
            return cls.dg(x)

        return self.evaluate_for_each_response(x, dg_of_x)

    def ddg(self, x):
        def ddg_of_x(cls, x):
            return cls.ddg(x)

        return self.evaluate_for_each_response(x, ddg_of_x)

    def update(self, x0, dg0, ddg0=0):
        for mp in self.map:
            mp[0].update(x0, dg0, ddg0=0)
        return self

    def clip(self, x):
        for mp in self.map:
            mp[0].clip(x)
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


class PositiveNegative(TwoMap, ABC):
    def __init__(self, left: Mapping, right: Mapping):
        super().__init__(left, right)
        self.positive = None

    def update(self, x0, dg0, ddg0=0):
        super().update(x0, dg0, ddg0)
        self.positive = dg0 >= 0

    def g(self, x): return np.where(self.positive, self.right.g(x), self.left.g(x))

    def dg(self, x): return np.where(self.positive, self.right.dg(x), self.left.dg(x))

    def ddg(self, x): return np.where(self.positive, self.right.ddg(x), self.left.ddg(x))
