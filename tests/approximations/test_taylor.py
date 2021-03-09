import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.approximations.intervening import ConLin


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor1(n, h):

    print("Testing 1st-order Taylor expansion with y=x")
    prob = Square(n)
    taylor1 = Taylor1()
    taylor1.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))

    # Check validity of Taylor1 expansion at expansion point X^(k)
    assert taylor1.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert taylor1.dg(prob.x0, np.ones_like(prob.x0)) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert taylor1.ddg(prob.x0, np.ones_like(prob.x0), np.zeros_like(prob.x0)) == pytest.approx(0, abs=1e-4)

    # Check validity of Taylor1 expansion at X^(k) + h
    assert taylor1.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + np.dot(prob.dg(prob.x0), h*np.ones_like(prob.x0)), rel=1e-4)
    assert taylor1.dg(prob.x0 + h, np.ones_like(prob.x0 + h)) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert taylor1.ddg(prob.x0 + h, np.ones_like(prob.x0 + h), np.zeros_like(prob.x0 + h)) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor2(n, h):
    print("Testing 2nd-order Taylor expansion with y=x")
    prob = Square(n)
    taylor2 = Taylor2()
    taylor2.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0), prob.ddg(prob.x0))

    # Check validity of Taylor2 expansion at expansion point X^(k)
    assert taylor2.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert taylor2.dg(prob.x0, np.ones_like(prob.x0)) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert taylor2.ddg(prob.x0, np.ones_like(prob.x0), np.zeros_like(prob.x0)) == pytest.approx(prob.ddg(prob.x0), rel=1e-4)

    # Check validity of Taylor2 expansion at X^(k) + h
    assert taylor2.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + np.dot(prob.dg(prob.x0), h*np.ones_like(prob.x0)) +
                                                  0.5 * np.dot(prob.ddg(prob.x0), h**2*np.ones_like(prob.x0)), rel=1e-4)
    assert taylor2.dg(prob.x0 + h, np.ones_like(prob.x0)) == pytest.approx(prob.dg(prob.x0) + prob.ddg(prob.x0) * h, rel=1e-4)
    assert taylor2.ddg(prob.x0 + h, np.ones_like(prob.x0), np.zeros_like(prob.x0)) == pytest.approx(prob.ddg(prob.x0) * np.ones_like(prob.x0)**2, rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor1_intervening(n, h):

    print("Testing 1st-order Taylor expansion with y=ConLin")
    prob = Square(n)
    inter = ConLin()
    inter.update(prob.x0, prob.dg(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)
    taylor1 = Taylor1()
    taylor1.update(inter.y(prob.x0).T, prob.g(prob.x0), dfdy)

    # Check validity of Taylor expansion at expansion point X^(k)
    assert taylor1.g(inter.y(prob.x0).T) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert taylor1.dg(inter.y(prob.x0).T, inter.dydx(prob.x0)) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert taylor1.ddg(inter.y(prob.x0).T, inter.dydx(prob.x0), inter.ddyddx(prob.x0)) == pytest.approx(dfdy * inter.ddyddx(prob.x0), abs=1e-4)

    # Check validity of Taylor expansion at X^(k) + h
    delta_y = (inter.y(prob.x0 + h) - inter.y(prob.x0)).T
    assert taylor1.g(inter.y(prob.x0 + h).T) == pytest.approx(prob.g(prob.x0) + np.diag(dfdy.dot(delta_y)), rel=1e-4)
    assert taylor1.dg(inter.y(prob.x0 + h).T, inter.dydx(prob.x0 + h)) == pytest.approx(dfdy * inter.dydx(prob.x0 + h), rel=1e-4)
    assert taylor1.ddg(inter.y(prob.x0 + h).T, inter.dydx(prob.x0 + h), inter.ddyddx(prob.x0 + h)) == pytest.approx(dfdy * inter.ddyddx(prob.x0 + h), rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor2_intervening(n, h):
    print("Testing 2nd-order Taylor expansion with y=ConLin")
    prob = Square(n)
    inter = ConLin()
    inter.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)
    ddfddy = prob.ddg(prob.x0) * (inter.dxdy(prob.x0)) ** 2 + prob.dg(prob.x0) * (inter.ddxddy(prob.x0))
    taylor2 = Taylor2(force_convex=False)
    taylor2.update(inter.y(prob.x0).T, prob.g(prob.x0), dfdy, ddfddy)

    # Check validity of Taylor expansion at expansion point X^(k)
    assert taylor2.g(inter.y(prob.x0).T) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert taylor2.dg(inter.y(prob.x0).T, inter.dydx(prob.x0)) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert taylor2.ddg(inter.y(prob.x0).T, inter.dydx(prob.x0), inter.ddyddx(prob.x0)) == pytest.approx(prob.ddg(prob.x0), rel=1e-4)

    # Check validity of Taylor expansion at X^(k) + h
    delta_y = (inter.y(prob.x0 + h) - inter.y(prob.x0)).T
    assert taylor2.g(inter.y(prob.x0 + h).T) == pytest.approx(prob.g(prob.x0) + np.diag(dfdy.dot(delta_y)) +
                                                             + 0.5 * np.diag(ddfddy.dot(delta_y**2)), rel=1e-4)
    assert taylor2.dg(inter.y(prob.x0 + h).T, inter.dydx(prob.x0 + h)) == pytest.approx(dfdy * inter.dydx(prob.x0 + h) +
                                                                                    ddfddy * delta_y.T * inter.dydx(prob.x0 + h), rel=1e-4)
    assert taylor2.ddg(inter.y(prob.x0 + h).T, inter.dydx(prob.x0 + h), inter.ddyddx(prob.x0 + h)) == pytest.approx(dfdy * inter.ddyddx(prob.x0 + h) +
                                                                                                            ddfddy * delta_y.T * inter.ddyddx(prob.x0 + h) +
                                                                                                            ddfddy * inter.dydx(prob.x0 + h) ** 2, rel=1e-4)


if __name__ == "__main__":
    test_taylor1(4, 0.1)
    test_taylor2(4, 0.1)
    test_taylor1_intervening(4, 0.1)
    test_taylor2_intervening(4, 0.1)
