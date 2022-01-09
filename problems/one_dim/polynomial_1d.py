import numpy as np
from sao.problems import Problem


class Polynomial1D(Problem):
    """
    This class is a 1D problem with a polynomial objective function and a polynomial constraint that is used to validate
    multi-point Taylor-like expansions. Both functions have a convex and a concave part. Moreover, the objective has 2
    local minima.
    """

    def __init__(self):
        super().__init__()
        self.n = 1
        self.m = 1
        self.x_min = np.array([-2.])
        self.x_max = np.array([1.5])
        self.x0 = np.array([1.2])  # use 6.9 to avoid local minimum or 11.5 to see if it gets stuck

    def g(self, x):
        g_j = np.empty(self.m + 1)
        g_j[0] = 4 - x[0] - 2 * x[0] ** 2 + 1.2 * x[0] ** 3 + x[0] ** 4
        g_j[1] = -1 - (x[0] - 0.5) ** 3
        return g_j

    def dg(self, x):
        dg_j = np.empty((self.m + 1, self.n))
        dg_j[0, 0] = -1 - 4 * x[0] + 3.6 * x[0] ** 2 + 4 * x[0] ** 3
        dg_j[1, 0] = -3 * (x[0] - 0.5) ** 2
        return dg_j

    def ddg(self, x):
        ddg_j = np.empty((self.m + 1, self.n))
        ddg_j[0, 0] = -4 + 7.2 * x[0] + 12 * x[0] ** 2
        ddg_j[1, 0] = -6 * (x[0] - 0.5)
        return ddg_j


if __name__ == "__main__":
    from problems.util.fd import finite_difference

    problem = Polynomial1D()
    finite_difference(problem, problem.x0, 1e-7)
