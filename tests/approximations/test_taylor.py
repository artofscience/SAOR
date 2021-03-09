import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.approximations.intervening import ConLin


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor1(n, h):

    print("Testing 1st-order Taylor expansion")
    prob = Square(n)
    taylor1 = Taylor1()
    taylor1.update(prob.x, prob.g(prob.x), prob.dg(prob.x))

    # Check validity of Taylor expansion at expansion point X^(k)
    assert taylor1.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert taylor1.dg(prob.x, np.ones_like(prob.x)) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert taylor1.ddg(prob.x, np.ones_like(prob.x), np.zeros_like(prob.x)) == pytest.approx(0, abs=1e-4)

    # Check validity of Taylor expansion at a point close to the expansion point X^(k)
    assert taylor1.g(prob.x + h) == pytest.approx(prob.g(prob.x) + np.dot(prob.dg(prob.x), h*np.ones_like(prob.x)), rel=1e-4)
    assert taylor1.dg(prob.x + h, np.ones_like(prob.x + h)) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert taylor1.ddg(prob.x + h, np.ones_like(prob.x + h), np.zeros_like(prob.x + h)) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor2(n, h):
    print("Testing 2nd-order Taylor expansion")
    prob = Square(n)
    taylor2 = Taylor2()
    taylor2.update(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    # Check validity of Taylor expansion at expansion point X^(k)
    assert taylor2.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert taylor2.dg(prob.x, np.ones_like(prob.x)) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert taylor2.ddg(prob.x, np.ones_like(prob.x), np.zeros_like(prob.x)) == pytest.approx(prob.ddg(prob.x), rel=1e-4)

    # Check validity of Taylor expansion at a point close to the expansion point X^(k)
    assert taylor2.g(prob.x + h) == pytest.approx(prob.g(prob.x) + np.dot(prob.dg(prob.x), h*np.ones_like(prob.x)) +
                                                  0.5 * np.dot(prob.ddg(prob.x), h**2*np.ones_like(prob.x)), rel=1e-4)
    assert taylor2.dg(prob.x + h, np.ones_like(prob.x)) == pytest.approx(prob.dg(prob.x) + prob.ddg(prob.x) * h, rel=1e-4)
    assert taylor2.ddg(prob.x + h, np.ones_like(prob.x), np.zeros_like(prob.x)) == pytest.approx(prob.ddg(prob.x) * np.ones_like(prob.x)**2, rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor1_intervening(n, h):

    print("Testing 1st-order Taylor expansion")
    prob = Square(n)
    inter = ConLin()
    inter.update(prob.x, prob.dg(prob.x), prob.dg(prob.x))
    dfdy = prob.dg(prob.x) * inter.dxdy(prob.x)
    taylor1 = Taylor1()
    taylor1.update(inter.y(prob.x).T, prob.g(prob.x), dfdy)

    # Check validity of Taylor expansion at expansion point X^(k)
    assert taylor1.g(inter.y(prob.x).T) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert taylor1.dg(inter.y(prob.x).T, inter.dy(prob.x)) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert taylor1.ddg(inter.y(prob.x).T, inter.dy(prob.x), inter.ddy(prob.x)) == pytest.approx(dfdy * inter.ddy(prob.x), abs=1e-4)

    # Check validity of Taylor expansion at a point close to the expansion point X^(k)
    delta_y = (inter.y(prob.x + h) - inter.y(prob.x)).T
    assert taylor1.g(inter.y(prob.x + h).T) == pytest.approx(prob.g(prob.x) + np.diag(dfdy.dot(delta_y)), rel=1e-4)
    assert taylor1.dg(inter.y(prob.x + h).T, inter.dy(prob.x + h)) == pytest.approx(dfdy * inter.dy(prob.x + h), rel=1e-4)
    assert taylor1.ddg(inter.y(prob.x + h).T, inter.dy(prob.x + h), inter.ddy(prob.x + h)) == pytest.approx(dfdy * inter.ddy(prob.x + h), rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor2_intervening(n, h):
    print("Testing 2nd-order Taylor expansion")
    prob = Square(n)
    inter = ConLin()
    inter.update(prob.x, prob.g(prob.x), prob.dg(prob.x))
    dfdy = prob.dg(prob.x) * inter.dxdy(prob.x)
    ddfddy = prob.ddg(prob.x) * (inter.dxdy(prob.x)) ** 2 + prob.dg(prob.x) * (inter.ddxddy(prob.x))
    taylor2 = Taylor2(force_convex=False)
    taylor2.update(inter.y(prob.x).T, prob.g(prob.x), dfdy, ddfddy)

    # Check validity of Taylor expansion at expansion point X^(k)
    assert taylor2.g(inter.y(prob.x).T) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert taylor2.dg(inter.y(prob.x).T, inter.dy(prob.x)) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert taylor2.ddg(inter.y(prob.x).T, inter.dy(prob.x), inter.ddy(prob.x)) == pytest.approx(prob.ddg(prob.x), rel=1e-4)

    # Check validity of Taylor expansion at a point close to the expansion point X^(k)
    delta_y = (inter.y(prob.x + h) - inter.y(prob.x)).T
    assert taylor2.g(inter.y(prob.x + h).T) == pytest.approx(prob.g(prob.x) + np.diag(dfdy.dot(delta_y)) +
                                                             + 0.5 * np.diag(ddfddy.dot(delta_y**2)), rel=1e-4)
    assert taylor2.dg(inter.y(prob.x + h).T, inter.dy(prob.x + h)) == pytest.approx(dfdy * inter.dy(prob.x + h) +
                                                                                    ddfddy * delta_y.T * inter.dy(prob.x + h), rel=1e-4)
    assert taylor2.ddg(inter.y(prob.x + h).T, inter.dy(prob.x + h), inter.ddy(prob.x + h)) == pytest.approx(dfdy * inter.ddy(prob.x + h) +
                                                                                                            ddfddy * delta_y.T * inter.ddy(prob.x + h) +
                                                                                                            ddfddy * inter.dy(prob.x + h) ** 2, rel=1e-4)


if __name__ == "__main__":
    test_taylor1(4, 0.1)
    test_taylor2(4, 0.1)
    test_taylor1_intervening(4, 0.1)
    test_taylor2_intervening(4, 0.1)

