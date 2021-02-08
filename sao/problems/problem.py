import numpy as np


class Problem(object):
    """A generic constrained optimisation problem."""
    def __init__(self, objective, constraints=[], bounds=(-np.inf, np.inf)):

        assert np.all([
            objective.n == c.n for c in constraints
        ]), "Objective and constraints dimensionality are not equal"

        self.objective = objective
        self.constraints = constraints
        self.bounds = bounds

    def response(self, x):
        """Returns the response of the objective and constraints at `x`."""
        r = [self.objective.f(x)] + [c.f(x) for c in self.constraints]
        return np.asarray(r)

    def sensitivity(self, x):
        """Returns the sensitivity of the objective and constraints at `x`."""
        dr = [self.objective.df(x)] + [c.df(x) for c in self.constraints]
        return np.asarray(dr)
