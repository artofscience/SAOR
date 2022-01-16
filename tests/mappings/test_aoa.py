from sao.problems import Problem
import numpy as np
from sao.mappings.mapping import Taylor2 as T2
from sao.mappings.mapping import Taylor1 as T1
from sao.mappings.mapping import Exponential as Exp
import pytest


class Dummy(Problem):
    def __init__(self, n):
        self.n = n
        self.x0 = np.linspace(1.0, 2.0, self.n, dtype=float)

    def g(self, x): return x @ x

    def dg(self, x): return 2 * x

    def ddg(self, x): return 2


def test_aoa_rec(dx=1, tol=1e-4):
    prob = Dummy(4)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)
    ddf = prob.ddg(x)

    rec = Exp(p=-1)
    rec.update(x, df, ddf)

    assert rec.g(x) == pytest.approx(1 / x, tol)
    assert rec.dg(x) == pytest.approx(-1 / x ** 2, tol)
    assert rec.ddg(x) == pytest.approx(2 / x ** 3, tol)

    t1_rec = T1(Exp(p=-1))
    aoa = T2(T1(Exp(p=-1)))
    t1_rec.update(x, df, ddf)
    aoa.update(x, df, ddf)

    assert aoa.map.map.g(x) == pytest.approx(1 / x, tol)
    assert aoa.map.map.dg(x) == pytest.approx(-1 / x ** 2, tol)
    assert aoa.map.map.ddg(x) == pytest.approx(2 / x ** 3, tol)

    y = x + dx

    assert aoa.map.map.g(y) == pytest.approx(1 / y, tol)
    assert aoa.map.map.dg(y) == pytest.approx(-1 / y ** 2, tol)
    assert aoa.map.map.ddg(y) == pytest.approx(2 / y ** 3, tol)

    assert t1_rec.g(y) == pytest.approx(x * df - x ** 2 * df / y, tol)
    assert t1_rec.dg(y) == pytest.approx(df * (x ** 2 / y ** 2))
    assert t1_rec.ddg(y) == pytest.approx(-2 * df * (x ** 2 / y ** 3))

    assert aoa.map.g(y) == pytest.approx(x * df - x ** 2 * df / y, tol)
    assert aoa.map.dg(y) == pytest.approx(df * (x ** 2 / y ** 2))
    assert aoa.map.ddg(y) == pytest.approx(-2 * df * (x ** 2 / y ** 3))


if __name__ == "__main__":
    test_aoa_rec()
