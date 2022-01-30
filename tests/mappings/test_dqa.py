from sao.problems import Problem
from sao.mappings.mapping import LinearApproximation as LA
from sao.mappings.mapping import DiagonalQuadraticApproximation as DQA
from sao.mappings.mapping import Exponential as Exp
import numpy as np
import pytest


class Dummy(Problem):
    def __init__(self, n):
        super().__init__()
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

    t1_rec = LA(Exp(p=-1))
    t1_rec.update(x, df)

    aoa = DQA()
    aoa.update(x, df, ddg0=t1_rec.ddg(x))

    assert aoa.g(x) == pytest.approx(t1_rec.g(x), tol)
    assert (f + np.sum(aoa.g(x))) == pytest.approx(f, tol)
    assert aoa.dg(x) == pytest.approx(t1_rec.dg(x), tol)
    assert aoa.dg(x) == pytest.approx(df, tol)
    assert aoa.ddg(x) == pytest.approx(df / t1_rec.map.dg(x) * t1_rec.map.ddg(x), tol)

    y = x + dx

    assert aoa.ddg(y) == pytest.approx(aoa.ddg(x), tol)
    assert aoa.dg(y) == pytest.approx(df + t1_rec.ddg(x) * (y - x), tol)
    assert aoa.g(y) == pytest.approx(df * (y - x) + 0.5 * t1_rec.ddg(x) * (y - x) ** 2)
    assert (f + np.sum(aoa.g(y))) == pytest.approx(f + np.sum(df * (y - x) + 0.5 * t1_rec.ddg(x) * (y - x) ** 2), tol)


def non_spherical(delta_dg, delta_x): return delta_dg / delta_x


def test_non_spherical(dx=1, tol=1e-4):
    prob = Dummy(4)
    x0 = prob.x0
    df0 = prob.dg(x0)

    aoa = DQA()
    aoa.update(x0, df0)

    x1 = x0 + dx

    df1 = prob.dg(x1)
    aoa.update(x1, df1, ddg0=non_spherical(df0 - df1, x0 - x1))

    assert aoa.ddg0 == pytest.approx((df0 - df1) / (x0 - x1), tol)
    assert aoa.dg(x0) == pytest.approx(df0, tol)


def spherical(delta_g, delta_x, dg): return 2 * (delta_g - dg @ delta_x) / np.sum(delta_x ** 2)


def test_spherical(dx=1, tol=1e-4):
    prob = Dummy(4)
    x0 = prob.x0
    f0 = prob.g(x0)
    df0 = prob.dg(x0)

    aoa = DQA()
    aoa.update(x0, df0)

    x1 = x0 + dx

    f1 = prob.g(x1)
    df1 = prob.dg(x1)

    aoa.update(x1, df1, ddg0=spherical(f0 - f1, x0 - x1, df1))

    assert aoa.ddg0 == pytest.approx(2 * (f0 - f1 - df1 @ (x0 - x1)) / np.sum((x0 - x1) ** 2), tol)
    assert (f1 + np.sum(aoa.g(x0))) == pytest.approx(f0, tol)


if __name__ == "__main__":
    test_aoa_rec()
    test_non_spherical()
    test_spherical()
