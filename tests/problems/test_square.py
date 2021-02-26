import pytest
import numpy as np
from sao.solvers.interior_point_basis import InteriorPointBasis as ipb


@pytest.mark.parametrize('n', [10, 1000, 100000])
def test_square(n):
    problem = Square(n)
    mysolver = ipb(problem, epsimin=1e-9)
    mysolver.update()
    assert np.sum(mysolver.w[0]) == pytest.approx(1, rel=1e-4)


class Square:
    """
    min(x)  x.x
    s.t.    sum(x) < 0.1
            -1 < x_i < 1,   i=1,...,n

    Solution: all x = 0.1/n
    """
    def __init__(self, n):
        self.alpha = -np.ones(n)
        self.beta = np.ones(n)
        self.x = np.random.uniform(self.alpha, self.beta)
        self.n = n
        self.m = 1
        self.zo = np.empty(self.m+1)
        self.zo[0] = 0
        self.zo[1] = -1

    def g(self, x):
        return np.array([np.dot(x, x), -np.sum(x)])

    def dg(self, x):
        return np.array([2*x, -np.ones_like(x)])

    def ddg(self, x):
        return np.array([2*np.ones_like(x), np.zeros_like(x)])


if __name__ == "__main__":
    test_square(100000)
