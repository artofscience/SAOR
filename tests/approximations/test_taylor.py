import pytest
import numpy as np
import logging
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.intervening_vars.intervening import Linear, ConLin

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor1(n, h):

    logger.info("Testing 1st-order Taylor expansion with y=x")
    prob = Square(n)
    inter = Linear()
    inter.update(prob.x0, prob.dg(prob.x0), prob.dg(prob.x0))
    taylor1 = Taylor1()
    taylor1.update(prob.x0, inter.y, prob.g(prob.x0), prob.dg(prob.x0), inter.dxdy)

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
    logger.info("Testing 2nd-order Taylor expansion with y=x")
    prob = Square(n)
    inter = Linear()
    inter.update(prob.x0, prob.dg(prob.x0), prob.dg(prob.x0))
    taylor2 = Taylor2()
    taylor2.update(prob.x0, inter.y, prob.g(prob.x0), prob.dg(prob.x0), inter.dxdy, ddf=prob.ddg(prob.x0), ddxddy=inter.ddxddy)

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

    logger.info("Testing 1st-order Taylor expansion with y=ConLin")
    prob = Square(n)
    inter = ConLin()
    inter.update(prob.x0, prob.dg(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)
    taylor1 = Taylor1()
    taylor1.update(prob.x0, inter.y, prob.g(prob.x0), prob.dg(prob.x0), inter.dxdy)

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
    logger.info("Testing 2nd-order Taylor expansion with y=ConLin")
    prob = Square(n)
    inter = ConLin()
    inter.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)
    ddfddy = prob.ddg(prob.x0) * (inter.dxdy(prob.x0)) ** 2 + prob.dg(prob.x0) * (inter.ddxddy(prob.x0))
    taylor2 = Taylor2(force_convex=False)
    taylor2.update(prob.x0, inter.y, prob.g(prob.x0), prob.dg(prob.x0), inter.dxdy, ddf=prob.ddg(prob.x0), ddxddy=inter.ddxddy)

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


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_SphericalTaylor2(n, h):
    logger.info("Testing SphericalTaylor2 expansion with y=x & y=ConLin")
    prob = Square(n)
    inter_list = [Linear(), ConLin()]
    for inter in inter_list:
        inter.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
        dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)
        sph_taylor2 = SphericalTaylor2(force_convex=False)
        sph_taylor2.update(prob.x0, inter.y, prob.g(prob.x0), prob.dg(prob.x0), inter.dxdy)

        # Check validity of SphericalTaylor2 expansion at expansion point X^(k)
        assert sph_taylor2.g(inter.y(prob.x0).T) == pytest.approx(prob.g(prob.x0), rel=1e-4)
        assert sph_taylor2.dg(inter.y(prob.x0).T, inter.dydx(prob.x0)) == pytest.approx(prob.dg(prob.x0), rel=1e-4)

        # Check validity of Taylor expansion at x1 = x0 + h
        x1 = prob.x0 + h
        delta_y = (inter.y(x1) - inter.y(prob.x0)).T
        if len(delta_y.shape) == 1:
            assert sph_taylor2.g(inter.y(x1).T) == pytest.approx(prob.g(prob.x0) + dfdy.dot(delta_y) +
                                                                 + 0.5 * (sph_taylor2.ddfddy.dot(delta_y ** 2)),
                                                                 rel=1e-4)
        else:
            assert sph_taylor2.g(inter.y(x1).T) == pytest.approx(prob.g(prob.x0) + np.diag(dfdy.dot(delta_y)) +
                                                                 + 0.5 * np.diag(sph_taylor2.ddfddy.dot(delta_y ** 2)),
                                                                 rel=1e-4)
        assert sph_taylor2.dg(inter.y(x1).T, inter.dydx(prob.x0 + h)) == pytest.approx(
            dfdy * inter.dydx(x1) + sph_taylor2.ddfddy * delta_y.T * inter.dydx(x1), rel=1e-4)
        assert sph_taylor2.ddg(inter.y(x1).T, inter.dydx(x1), inter.ddyddx(x1)) == pytest.approx(
            dfdy * inter.ddyddx(x1) + sph_taylor2.ddfddy * delta_y.T * inter.ddyddx(x1) +
            sph_taylor2.ddfddy * inter.dydx(x1) ** 2, rel=1e-4)

        # Check if previous point is satisfied
        inter.update(x1, prob.g(x1), prob.dg(x1))
        sph_taylor2.update(x1, inter.y, prob.g(x1), prob.dg(x1), inter.dxdy)
        assert sph_taylor2.g(inter.y(sph_taylor2.xold1).T) == pytest.approx(prob.g(prob.x0), rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_NonSphericalTaylor2(n, h):
    logger.info("Testing NonSphericalTaylor2 expansion with y=x & y=ConLin")
    prob = Square(n)
    inter_list = [Linear(), ConLin()]
    for inter in inter_list:
        inter.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
        dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)
        nsph_taylor2 = NonSphericalTaylor2(force_convex=False)
        nsph_taylor2.update(prob.x0, inter.y, prob.g(prob.x0), prob.dg(prob.x0), inter.dxdy)

        # Check validity of Taylor expansion at expansion point X^(k)
        assert nsph_taylor2.g(inter.y(prob.x0).T) == pytest.approx(prob.g(prob.x0), rel=1e-4)
        assert nsph_taylor2.dg(inter.y(prob.x0).T, inter.dydx(prob.x0)) == pytest.approx(prob.dg(prob.x0), rel=1e-4)

        # Check validity of NonSphericalTaylor2 expansion at x1 = x0 + h
        x1 = prob.x0 + h
        delta_y = (inter.y(x1) - inter.y(prob.x0)).T
        if len(delta_y.shape) == 1:
            assert nsph_taylor2.g(inter.y(x1).T) == pytest.approx(prob.g(prob.x0) + dfdy.dot(delta_y) +
                                                                  + 0.5 * (nsph_taylor2.ddfddy.dot(delta_y ** 2)),
                                                                  rel=1e-4)
        else:
            assert nsph_taylor2.g(inter.y(x1).T) == pytest.approx(prob.g(prob.x0) + np.diag(dfdy.dot(delta_y)) +
                                                                  + 0.5 * np.diag(nsph_taylor2.ddfddy.dot(delta_y ** 2)),
                                                                  rel=1e-4)
        assert nsph_taylor2.dg(inter.y(x1).T, inter.dydx(x1)) == pytest.approx(
            dfdy * inter.dydx(x1) + nsph_taylor2.ddfddy * delta_y.T * inter.dydx(x1), rel=1e-4)
        assert nsph_taylor2.ddg(inter.y(x1).T, inter.dydx(x1), inter.ddyddx(x1)) == pytest.approx(
            dfdy * inter.ddyddx(x1) + nsph_taylor2.ddfddy * delta_y.T * inter.ddyddx(x1) +
            nsph_taylor2.ddfddy * inter.dydx(x1) ** 2, rel=1e-4)

        # Check if previous point is satisfied
        inter.update(x1, prob.g(x1), prob.dg(x1))
        nsph_taylor2.update(x1, inter.y, prob.g(x1), prob.dg(x1), inter.dxdy)
        assert nsph_taylor2.dg(inter.y(nsph_taylor2.xold1).T, inter.dydx(nsph_taylor2.xold1)) == pytest.approx(
            prob.dg(prob.x0), rel=1e-4)


if __name__ == "__main__":
    test_taylor1(4, 0.1)
    test_taylor2(4, 0.1)
    test_taylor1_intervening(4, 0.1)
    test_taylor2_intervening(4, 0.1)
    test_SphericalTaylor2(4, 0.1)
    test_NonSphericalTaylor2(4, 0.1)
