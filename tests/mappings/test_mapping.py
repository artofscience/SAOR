from problems.n_dim.square import Square
from sao.mappings.intervening import Exponential as Exp
from sao.mappings.approximations import DiagonalQuadraticApproximation as DQA
import pytest
from sao.mappings.mapping import MixedMapping as MM


def test_mm(dx=1, tol=1e-4):
    prob = Square(2)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)
    ddf = prob.ddg(x)

    mymap = MM(prob.n, prob.m + 1)
    mymap.set_map(Exp(p=-1), 0, 1)
    mymap.set_map(Exp(p=-2), 1, 1)
    mymap.update(prob.x0, df, ddf)

    aoa = DQA()
    aoa.update(x, df, ddg0=mymap.ddg(x))

    assert mymap.g(x)[0, 0] == pytest.approx(x[0], tol)
    assert mymap.g(x)[1, 0] == pytest.approx(x[0], tol)
    assert mymap.g(x)[0, 1] == pytest.approx(1 / x[1], tol)
    assert mymap.g(x)[1, 1] == pytest.approx(1 / (x[1]) ** 2, tol)

    assert aoa.ddg0 == pytest.approx(mymap.ddg(x), tol)


if __name__ == "__main__":
    test_mm()
