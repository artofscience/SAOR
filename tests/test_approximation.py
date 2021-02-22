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

    assert taylor.g_approx(x) == f(x)
    assert taylor.dg_approx(x) == df(x)
    assert taylor.ddg_approx(x) == 1.0
