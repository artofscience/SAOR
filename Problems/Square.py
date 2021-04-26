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
        self.xmin = self.alpha = - np.ones(n)              # cuz a subproblem uses both, whereas a problem only has xmin
        self.xmax = self.beta = np.ones(n)                 # cuz a subproblem uses both, whereas a problem only has xmax
        self.x0 = np.linspace(0.5, 0.9, n)
        # self.x0 = np.random.rand(self.alpha, self.beta, n)
        self.n = n
        self.m = 1

    def g(self, x):
        return np.array([np.dot(x, x), 1-np.sum(x)])

    def dg(self, x):
        return np.array([2*x, -np.ones_like(x)])

    def ddg(self, x):
        return np.array([2*np.ones_like(x), np.zeros_like(x)])
