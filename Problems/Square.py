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

        # Allocated vector for the response and derivatives, subsequent
        # interations can reuse the storage.
        self._g = None
        self._dg = None
        self._ddg = None

    def g(self, x):
        if self._g is None:
            self._g = np.array([np.dot(x, x), 1-np.sum(x)])
        else:
            self._g[0] = np.dot(x, x)
            self._g[1] = 1 - np.sum(x)
        return self._g

    def dg(self, x):
        if self._dg is None:
            self._dg = np.array([2*x, -np.ones_like(x)])
        else:
            self._dg[0] = 2*x
            self._dg[1] = -1
        return self._dg

    def ddg(self, x):
        if self._ddg is None:
            self._ddg = np.array([2*np.ones_like(x), np.zeros_like(x)])
        else:
            self._ddg[0] = 2
            self._ddg[1] = 0
        return self._ddg

