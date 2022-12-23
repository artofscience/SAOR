from problems.n_dim.square import Square
from sao.mappings.change_of_variable import Exponential as Exp
from sao.mappings.approximations import DiagonalQuadraticApproximation as DQA
import pytest
from sao.mappings.mapping import MixedMapping as MM
import numpy as np


def test_mmn(dx=1, tol=1e-4):
    prob = Square(3)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)
    ddf = prob.ddg(x)

    mm = MM(prob.n, prob.m + 1)
    mm[np.array([0]), [0, 1, 2]] = Exp(-1)
    mm[[1], [2]] = Exp(-2)

    mm.update(prob.x0, df, ddf)

    aoa = DQA()
    aoa.update(x, df, ddg0=mm.ddg(x))

    assert mm.g(x)[1, [0, 1]] == pytest.approx(x[0:2], tol)
    assert mm.g(x)[0, [0, 1, 2]] == pytest.approx(1 / x, tol)
    assert mm.g(x)[1, 2] == pytest.approx(1 / (x[2]) ** 2, tol)

    assert aoa.ddg0 == pytest.approx(mm.ddg(x), tol)


if __name__ == "__main__":
    test_mmn()