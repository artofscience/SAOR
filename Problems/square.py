import numpy as np


class Square:
    """
    min(x)  x.x
    s.t.    sum(x) > 1
            -1 < x_i < 1,   i=1,...,n

    Solution: all x = 1/n
    """
    def __init__(self, n):
        self.xmin = self.alpha = -np.ones(n)                # cuz approx needs to have both
        self.xmax = self.beta = np.ones(n)                  # cuz approx needs to have both
        self.x = np.random.uniform(self.xmin, self.xmax)
        self.n = n
        self.m = 1

    def g(self, x):
        return np.array([np.dot(x, x), 1-np.sum(x)])

    def dg(self, x):
        return np.array([2*x, -np.ones_like(x)])

    def ddg(self, x):
        return np.array([2*np.ones_like(x), np.zeros_like(x)])
