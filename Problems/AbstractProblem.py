import numpy as np


class Problem:
    """A generic constrained optimization problem."""
    def __init__(self, response_funcs, xmin, xmax, **kwargs):
        self.response_funcs = response_funcs
        self.m = len(response_funcs) - 1            # objective + constraints
        self.xmin = xmin
        self.xmax = xmax
        self.x_init = kwargs.get('x_init', 0.5 * (self.xmin + self.xmax))

    def response(self, x):
        """Returns the response of the objective and constraints at `x`."""
        g_j = [c.f(x) for c in self.response_funcs]
        return np.asarray(g_j)

    def sensitivity(self, x):
        """Returns the sensitivity of the objective and constraints at `x`."""
        dg_j = [c.df(x) for c in self.response_funcs]
        return np.asarray(dg_j)
