import logging

import numpy as np
import pytest

from problems.n_dim.square import Square
from sao.approximations.taylor import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.intervening_variables import Linear, ConLin

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
    problem = Square(n)
    f = problem.g(problem.x0)
    df = problem.dg(problem.x0)

    taylor1 = Taylor1(Linear())
    taylor1.update(problem.x0, f, df)

    # Check validity of Taylor expansion at X^(k)
    assert taylor1.g(problem.x0) == pytest.approx(f, rel=1e-4)
    assert taylor1.dg(problem.x0) == pytest.approx(df, rel=1e-4)
    assert taylor1.ddg(problem.x0) == pytest.approx(0, abs=1e-4)

    # Check validity of Taylor expansion at X^(k) + dX
    dx = np.ones_like(problem.x0) * h
    assert taylor1.g(problem.x0 + dx) == pytest.approx(f + df@dx, rel=1e-4)
    assert taylor1.dg(problem.x0 + h) == pytest.approx(problem.dg(problem.x0), rel=1e-4)
    assert taylor1.ddg(problem.x0 + h) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor2(n, h):
    logger.info("Testing 2nd-order Taylor expansion with y=x")
    problem = Square(n)
    f = problem.g(problem.x0)
    df = problem.dg(problem.x0)
    ddf = problem.ddg(problem.x0)
    taylor2 = Taylor2(Linear())
    taylor2.update(problem.x0, f, df, ddf)

    # Check validity of Taylor expansion at X^(k)
    assert taylor2.g(problem.x0) == pytest.approx(f, rel=1e-4)
    assert taylor2.dg(problem.x0) == pytest.approx(df, rel=1e-4)
    assert taylor2.ddg(problem.x0) == pytest.approx(ddf, abs=1e-4)

    # Check validity of Taylor expansion at X^(k) + dX
    dx = np.ones_like(problem.x0) * h
    assert taylor2.g(problem.x0 + dx) == pytest.approx(f + df @ dx + 0.5 * ddf @ (dx**2), rel=1e-4)
    assert taylor2.dg(problem.x0 + dx) == pytest.approx(df + ddf*dx , rel=1e-4)
    assert taylor2.ddg(problem.x0 + dx) == pytest.approx(ddf, abs=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor1_intervening(n, h):
    logger.info("Testing 1st-order Taylor expansion with y=ConLin")
    problem = Square(n)
    inter = ConLin()
    f = problem.g(problem.x0)
    df = problem.dg(problem.x0)
    taylor1 = Taylor1(inter)
    taylor1.update(problem.x0, f, df)

    # Check validity of Taylor expansion at X^(k)
    assert taylor1.g(problem.x0) == pytest.approx(f, rel=1e-4)
    assert taylor1.dg(problem.x0) == pytest.approx(df, rel=1e-4)

    # Check validity of Taylor expansion at X^(k) + dX
    dx = np.ones_like(problem.x0) * h
    dfdy = df * inter.dxdy(problem.x0)
    dy = (inter.y(problem.x0 + dx) - inter.y(problem.x0)).T
    assert taylor1.g(problem.x0 + dx) == pytest.approx(f + np.diag(dfdy @ dy), rel=1e-4)
    assert taylor1.dg(problem.x0 + dx) == pytest.approx(dfdy * inter.dydx(problem.x0 + dx), rel=1e-4)
    assert taylor1.ddg(problem.x0 + dx) == pytest.approx(dfdy * inter.ddyddx(problem.x0 + dx), abs=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_taylor2_intervening(n, h):
    logger.info("Testing 2nd-order Taylor expansion with y=ConLin")
    problem = Square(n)
    inter = ConLin()
    f = problem.g(problem.x0)
    df = problem.dg(problem.x0)
    ddf = problem.ddg(problem.x0)
    taylor2 = Taylor2(inter)
    taylor2.update(problem.x0, f, df, ddf)

    # Check validity of Taylor expansion at X^(k)
    assert taylor2.g(problem.x0) == pytest.approx(f, rel=1e-4)
    assert taylor2.dg(problem.x0) == pytest.approx(df, rel=1e-4)
    assert taylor2.ddg(problem.x0) == pytest.approx(ddf, abs=1e-4)

    # Check validity of Taylor expansion at X^(k) + dX
    dx = np.ones_like(problem.x0) * h
    dfdy = df * inter.dxdy(problem.x0)
    ddfddy = ddf * (inter.dxdy(problem.x0))**2 + df * inter.ddxddy(problem.x0)
    dy = (inter.y(problem.x0 + dx) - inter.y(problem.x0)).T
    assert taylor2.g(problem.x0 + dx) == pytest.approx(f + np.diag(dfdy @ dy) + 0.5*np.diag(ddfddy @ dy**2), rel=1e-4)
    assert taylor2.dg(problem.x0 + dx) == pytest.approx(dfdy * inter.dydx(problem.x0 + dx) -
                                                        ddfddy * inter.y(problem.x0) * inter.dydx(problem.x0 + dx) +
                                                        ddfddy * inter.y(problem.x0 + dx) * inter.dydx(problem.x0 + dx)
                                                        , rel=1e-4)
    assert taylor2.ddg(problem.x0 + dx) == pytest.approx(dfdy * inter.ddyddx(problem.x0 + dx) -
                                                         ddfddy * inter.y(problem.x0) * inter.ddyddx(problem.x0 + dx) +
                                                         ddfddy * (inter.dydx(problem.x0 + dx))**2 +
                                                         ddfddy * inter.y(problem.x0 + dx) * inter.ddyddx(problem.x0+dx)
                                                         , abs=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_SphericalTaylor2_intervening(n, h):
    logger.info("Testing SphericalTaylor2 expansion with y=ConLin")
    problem = Square(n)
    inter = ConLin()
    f = problem.g(problem.x0)
    df = problem.dg(problem.x0)
    sph_taylor2 = SphericalTaylor2(inter)
    sph_taylor2.update(problem.x0, f, df)

    # Check validity of Taylor expansion at X^(0)
    assert sph_taylor2.g(problem.x0) == pytest.approx(f, rel=1e-4)
    assert sph_taylor2.dg(problem.x0) == pytest.approx(df, rel=1e-4)

    # Check validity of Taylor expansion at X^(0) + dX (at iter=0, SphericalTaylor2 == Taylor1 cuz you need prev. point)
    dx = np.ones_like(problem.x0) * h
    dfdy = df * inter.dxdy(problem.x0)
    ddfddy = df * inter.ddxddy(problem.x0)
    dy = (inter.y(problem.x0 + dx) - inter.y(problem.x0)).T
    assert sph_taylor2.g(problem.x0 + dx) == pytest.approx(f + np.diag(dfdy@dy) + 0.5*np.diag(ddfddy @ dy**2), rel=1e-4)
    assert sph_taylor2.dg(problem.x0 + dx) == pytest.approx(dfdy * inter.dydx(problem.x0 + dx) -
                                                            ddfddy * inter.y(problem.x0) * inter.dydx(problem.x0 + dx) +
                                                            ddfddy * inter.y(problem.x0+dx) * inter.dydx(problem.x0+dx)
                                                            , rel=1e-4)
    assert sph_taylor2.ddg(problem.x0 + dx) == pytest.approx(dfdy * inter.ddyddx(problem.x0 + dx) -
                                                             ddfddy * inter.y(problem.x0) * inter.ddyddx(problem.x0+dx) +
                                                             ddfddy * (inter.dydx(problem.x0 + dx)) ** 2 +
                                                             ddfddy * inter.y(problem.x0+dx) * inter.ddyddx(problem.x0+dx)
                                                             , abs=1e-4)

    # Check validity of Taylor expansion at the next iteration; here we use: X^(1) = X^(0) - dX
    x1 = problem.x0 - dx
    fold1 = f.copy()
    f = problem.g(x1)
    df = problem.dg(x1)
    sph_taylor2.update(x1, f, df)
    assert sph_taylor2.g(x1) == pytest.approx(f, rel=1e-4)
    assert sph_taylor2.dg(x1) == pytest.approx(df, rel=1e-4)

    # Check if previous point is satisfied
    assert sph_taylor2.g(problem.x0) == pytest.approx(fold1, rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_NonSphericalTaylor2_intervening(n, h):
    logger.info("Testing SphericalTaylor2 expansion with y=ConLin")
    problem = Square(n)
    inter = ConLin()
    f = problem.g(problem.x0)
    df = problem.dg(problem.x0)
    nsph_taylor2 = NonSphericalTaylor2(inter)
    nsph_taylor2.update(problem.x0, f, df)

    # Check validity of Taylor expansion at X^(0)
    assert nsph_taylor2.g(problem.x0) == pytest.approx(f, rel=1e-4)
    assert nsph_taylor2.dg(problem.x0) == pytest.approx(df, rel=1e-4)

    # Check validity of Taylor expansion at X^(0) + dX (at iter=0, NonSphericalTaylor2 == Taylor1)
    dx = np.ones_like(problem.x0) * h
    dfdy = df * inter.dxdy(problem.x0)
    ddfddy = df * inter.ddxddy(problem.x0)
    dy = (inter.y(problem.x0 + dx) - inter.y(problem.x0)).T
    assert nsph_taylor2.g(problem.x0 + dx) == pytest.approx(f + np.diag(dfdy @ dy) + 0.5 * np.diag(ddfddy @ dy ** 2),
                                                            rel=1e-4)
    assert nsph_taylor2.dg(problem.x0 + dx) == pytest.approx(dfdy * inter.dydx(problem.x0 + dx) -
                                                             ddfddy * inter.y(problem.x0) * inter.dydx(problem.x0 + dx) +
                                                             ddfddy * inter.y(problem.x0+dx) * inter.dydx(problem.x0+dx)
                                                             , rel=1e-4)
    assert nsph_taylor2.ddg(problem.x0 + dx) == pytest.approx(dfdy * inter.ddyddx(problem.x0 + dx) -
                                                              ddfddy * inter.y(problem.x0) * inter.ddyddx(problem.x0+dx) +
                                                              ddfddy * (inter.dydx(problem.x0 + dx)) ** 2 +
                                                              ddfddy * inter.y(problem.x0+dx) * inter.ddyddx(problem.x0+dx)
                                                              , abs=1e-4)

    # Check validity of Taylor expansion at the next iteration; here we use: X^(1) = X^(0) - dX
    x1 = problem.x0 - dx
    dfold1 = df.copy()
    f = problem.g(x1)
    df = problem.dg(x1)
    nsph_taylor2.update(x1, f, df)
    assert nsph_taylor2.g(x1) == pytest.approx(f, rel=1e-4)
    assert nsph_taylor2.dg(x1) == pytest.approx(df, rel=1e-4)

    # Check if previous point gradient is satisfied
    assert nsph_taylor2.dg(problem.x0) == pytest.approx(dfold1, rel=1e-4)


if __name__ == "__main__":
    test_taylor1(4, 0.1)
    test_taylor2(4, 0.1)
    test_taylor1_intervening(4, 0.1)
    test_taylor2_intervening(4, 0.1)
    test_SphericalTaylor2_intervening(4, 0.1)
    test_NonSphericalTaylor2_intervening(4, 0.1)