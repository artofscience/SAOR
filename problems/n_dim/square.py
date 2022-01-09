import numpy as np
from sao.problems.problem import Problem


class Square(Problem):
    """
    min(x)  x.x
    s.t.    sum(x) > 1
            -1 < x_i < 1,   i=1,...,n

    Solution: all x = 1/n
    """

    def __init__(self, n):
        super().__init__()
        self.x_min = 1e-3 * np.ones(n)  # cuz a subproblem uses both, whereas a problem only has x_min
        self.x_max = np.ones(n)  # cuz a subproblem uses both, whereas a problem only has x_max
        self.x0 = np.linspace(0.8, 0.9, n)
        self.n = n
        self.m = 1
        self.f = np.zeros(n)

    def g(self, x):
        self.f = np.array([np.dot(x, x), 1 - np.sum(x)])
        return self.f

    def dg(self, x):
        return np.array([2 * x, -np.ones_like(x)])

    def ddg(self, x):
        return np.array([2 * np.ones_like(x), np.zeros_like(x)])


if __name__ == "__main__":
    from problems.util.fd import finite_difference

    problem = Square(4)
    finite_difference(problem, problem.x0, 1e-7)
