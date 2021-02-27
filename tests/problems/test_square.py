import pytest
import numpy as np
from sao.solvers.interior_point_basis import InteriorPointBasis as ipb
from sao.solvers.interior_point_artificial import InteriorPointArtificial as ipa

@pytest.mark.parametrize('n', [10, 100, 1000])
def test_square(n):
    problem = Square(n)
    #
    # mysolverb = ipb(problem, epsimin=1e-7)
    # mysolverb.update()
    # print(np.sum(mysolverb.x))
    # assert np.sum(mysolverb.x) == pytest.approx(1, rel=1e-4)

    mysolvera = ipa(problem, epsimin=1e-5)
    mysolvera.update()
    print(np.sum(mysolvera.x))
    assert np.sum(mysolvera.x) == pytest.approx(1, rel=1e-4)


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

    def g(self, x):
        return np.array([np.dot(x, x), 1-np.sum(x)])

    def dg(self, x):
        return np.array([2*x, -np.ones_like(x)])

    def ddg(self, x):
        return np.array([2*np.ones_like(x), np.zeros_like(x)])

if __name__ == "__main__":
    test_square(4)
