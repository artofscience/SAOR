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


def fill_set_when_emtpy(s, n):
    """Returns ``set(s)`` or a ``set(0..n)`` if ``set(s)`` is the empty set."""
    if s is None or s is ...: return set(range(n))
    try:
        s = set(s)
    except TypeError:
        s = set([s])
    if len(s) == 0: return set(range(n))
    return s


class MixedMapping(Mapping):
    """
    For every response i, and variable j, a separate or combination of
    intervening variables can be set. (intervening, response, variable).

    The responses are tracked by sets of indices that indicate the response of
    interest for the given intervening variable. These sets are
    non-overlapping, i.e. there is only one intervening variable that points to
    a response at all times.

    The variables are tracked by a dictionary of response indices to variable
    sets. So, ``{0: {0, 1, 3}, 1: {0, 2}}`` indicates that for response ``0``
    the variables ``{0, 1, 3}`` are relevant and for response ``1`` only the
    variable set ``{0, 2}``. The variable sets used in different responses can
    overlap.
    """

    def __init__(self, nvar: int, nresp: int, default: Mapping = Linear()):
        super().__init__()
        self.default = default
        self.nvar = nvar
        self.nresp = nresp

        # On initialisation the default intervening variable is added to all
        # the responses pointing to all considered variables.
        responses = set(range(self.nresp))
        variables = set(range(self.nvar))
        self.add_map(self.default, responses, variables)

    @property
    def maps(self):
        """Yields only the intervening variables."""
        for mp, _, _ in self.map:
            yield mp

    def set_map(self, inter: Mapping, resp=set(), var=set()):
        """Assign a intervening variable to some variables/responses.

        Other intervening variables that might be pointing to the same
        responses are updated accordingly to avoid any overlap between the
        different response sets.
        """
        new_resp = fill_set_when_emtpy(resp, self.nresp)
        new_vars = fill_set_when_emtpy(var, self.nvar)

        for _, responses, variables in self.map:
            # Only consider to remove entries when the new response shares
            # the same indices as the existing responses (set intersection).
            for r in (new_resp & responses):
                diff = variables[r] - new_vars
                if len(diff) > 0:
                    # If the resulting set of variables is non-empty, we need
                    # to add the index `r` to the current set with the
                    # remaining variables.
                    responses.add(r)
                    variables[r] = diff
                else:
                    # If the resulting set is empty, the index `r` can be
                    # removed from the current set of responses and the
                    # corresponding variables can be deleted from the mapping.
                    responses.remove(r)
                    del variables[r]

        # After deleting the overlapping regions in any other response and/or
        # variable sets, an additional intervening variable is added.
        return self.add_map(inter, new_resp, new_vars)

    def add_map(self, inter, resp=set(), var=set()):
        """Adds an additional intervening variable to responses and variables.

        The mapping only considers the unique set of elements in the response
        and variable sets. When an empty is given, all responses/variables will
        be considered.
        """
        responses = fill_set_when_emtpy(resp, self.nresp)
        variables = fill_set_when_emtpy(var, self.nvar)

        self.map.append(
            (inter, responses, {i: variables for i in responses})
        )
        return self

    def evaluate_for_each_response(self, x, fn: callable):
        """Evaluates a function for each response and collects its output.

        Allocates the output of size ``number of reponses`` by ``number of
        design variables`` and populates the output by evaluating a callable
        function for each intervening variable given the current ``x``.
        """
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

    def y(self, x):
        """Evaluates the mapping y = f(x)."""

        def y_of_x(cls, x):
            return cls.y(x)

        return self.evaluate_for_each_response(x, y_of_x)

    def dydx(self, x):
        """Evaluates the first derivative of the mapping at x."""

        def dy_of_x(cls, x):
            return cls.dydx(x)

        return self.evaluate_for_each_response(x, dy_of_x)

    def ddyddx(self, x):
        """Evaluates the second derivatives of the mapping at x."""

        def ddy_of_x(cls, x):
            return cls.ddyddx(x)

        return self.evaluate_for_each_response(x, ddy_of_x)

    def update(self, x0, dg0, ddg0=0):
        for mp in self.map:
            mp.update(x0, dg0, ddg0=0)
        return self

    def clip(self, x):
        for mp in self.map:
            mp.clip(x)
        return x
