import numpy as np


class Problem:
    """A generic constrained optimization problem.
       responses = [objective, constraint_1, constraint_2, ..., constraint_m
       xmin = [x1_min, x2_min, ..., xn_min]
       xmax = [x1_max, x2_max, ..., xn_max]"""
    def __init__(self, responses, xmin, xmax, x0=None, **kwargs):
        self.responses = responses
        self.m = len(responses) - 1            # objective + constraints
        self.xmin = xmin
        self.xmax = xmax
        self.x0 = x0 if x0 is not None else 0.5 * (self.xmin + self.xmax)

    def response(self, x):
        """Returns the response of the objective and constraints at `x`."""
        return np.asarray([r.g(x) for r in self.responses])

    def sensitivity(self, x):
        """Returns the sensitivity of the objective and constraints at `x`."""
        return np.asarray([r.dg(x) for r in self.responses])
