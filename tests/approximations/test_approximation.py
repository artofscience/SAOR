import numpy as np

from sao.approximations.taylor import Taylor1


def test_taylor_1():
    n = 1
    x = np.ones(n)
    bounds = (0, 1)

    def f(x):
        return x

    def df(x):
        return np.ones((1, n))

    taylor = Taylor1()
    taylor.update_approximation(x, bounds, f(x), df(x))

    assert taylor.g(x) == f(x)
    assert taylor.dg(x) == df(x)
    assert taylor.ddg(x) == 1.0
