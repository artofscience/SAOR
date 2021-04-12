import pytest
import numpy as np
import logging
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.intervening_vars.intervening import Linear, Reciprocal, ConLin, MMA
from sao.move_limits.ml_intervening import MoveLimitIntervening
from sao.problems.subproblem import Subproblem

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
def test_lin_taylor1(n, h):
    logger.info("Testing Subproblem(Taylor1, y=x)")
    prob = Square(n)
    subprob = Subproblem(intervening=Linear(), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert subprob.dg(prob.x0) == pytest.approx(prob.dg(prob.x0), rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    assert subprob.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + np.dot(prob.dg(prob.x0), h*np.ones_like(prob.x0)), rel=1e-4)
    assert subprob.dg(prob.x0 + h) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert subprob.ddg(prob.x0 + h) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_lin_taylor2(n, h):
    logger.info("Subproblem(Taylor2, y=x)")
    prob = Square(n)
    subprob = Subproblem(intervening=Linear(), approximation=Taylor2(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x0, prob.g(prob.x0), prob.dg(prob.x0), prob.ddg(prob.x0))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert subprob.dg(prob.x0) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert subprob.ddg(prob.x0) == pytest.approx(prob.ddg(prob.x0), rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    assert subprob.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + np.dot(prob.dg(prob.x0), h*np.ones_like(prob.x0)) +
                                                  0.5 * np.dot(prob.ddg(prob.x0), h**2*np.ones_like(prob.x0)), rel=1e-4)
    assert subprob.dg(prob.x0 + h) == pytest.approx(prob.dg(prob.x0) + prob.ddg(prob.x0) * h, rel=1e-4)
    assert subprob.ddg(prob.x0 + h) == pytest.approx(prob.ddg(prob.x0) * np.ones_like(prob.x0)**2, rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_rec_taylor1(n, h):
    logger.info("Testing Subproblem(Taylor1, y=1/x)")
    prob = Square(n)
    subprob = Subproblem(intervening=Reciprocal(), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    inter = Reciprocal()
    inter.update(prob.x0, prob.dg(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * (-(prob.x0**2))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert subprob.dg(prob.x0) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert dfdy == pytest.approx(prob.dg(prob.x0) * inter.dxdy(prob.x0), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x0 + h) - inter.y(prob.x0)).T
    assert subprob.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + dfdy.dot(delta_y), rel=1e-4)
    assert subprob.dg(prob.x0 + h) == pytest.approx(dfdy * inter.dydx(prob.x0 + h), rel=1e-4)
    assert subprob.ddg(prob.x0 + h) == pytest.approx(dfdy * inter.ddyddx(prob.x0 + h), rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_rec_taylor2(n, h):
    logger.info("Testing Subproblem(Taylor2, y=1/x)")
    prob = Square(n)
    subprob = Subproblem(intervening=Reciprocal(), approximation=Taylor2(force_convex=False),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x0, prob.g(prob.x0), prob.dg(prob.x0), prob.ddg(prob.x0))
    inter = Reciprocal()
    inter.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * (-(prob.x0**2))
    ddfddy = prob.ddg(prob.x0) * prob.x0**4 + prob.dg(prob.x0) * 2 * prob.x0**3

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert subprob.dg(prob.x0) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.ddg(prob.x0) == pytest.approx(prob.ddg(prob.x0), rel=1e-4)
    assert dfdy == pytest.approx(prob.dg(prob.x0) * inter.dxdy(prob.x0), rel=1e-4)
    assert ddfddy == pytest.approx(prob.ddg(prob.x0) * (inter.dxdy(prob.x0)) ** 2 + prob.dg(prob.x0) * (inter.ddxddy(prob.x0)), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.approx.ddfddy == pytest.approx(ddfddy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x0 + h) - inter.y(prob.x0)).T
    assert subprob.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + dfdy.dot(delta_y) + 0.5 * ddfddy.dot(delta_y ** 2), rel=1e-4)
    assert subprob.dg(prob.x0 + h) == pytest.approx(dfdy * inter.dydx(prob.x0 + h) + ddfddy * delta_y.T * inter.dydx(prob.x0 + h), rel=1e-4)
    assert subprob.ddg(prob.x0 + h) == pytest.approx(dfdy * inter.ddyddx(prob.x0 + h) + ddfddy * delta_y.T *
                                                    inter.ddyddx(prob.x0 + h) + ddfddy * inter.dydx(prob.x0 + h) ** 2, rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_conlin_taylor1(n, h):
    logger.info("Testing Subproblem(Taylor1, y=ConLin)")
    prob = Square(n)
    subprob = Subproblem(intervening=ConLin(), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    inter = ConLin()
    inter.update(prob.x0, prob.dg(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert subprob.dg(prob.x0) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x0 + h) - inter.y(prob.x0)).T
    assert subprob.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + np.diag(dfdy.dot(delta_y)), rel=1e-4)
    assert subprob.dg(prob.x0 + h) == pytest.approx(dfdy * inter.dydx(prob.x0 + h), rel=1e-4)
    assert subprob.ddg(prob.x0 + h) == pytest.approx(dfdy * inter.ddyddx(prob.x0 + h), rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_conlin_taylor2(n, h):
    logger.info("Testing Subproblem(Taylor2, y=ConLin)")
    prob = Square(n)
    subprob = Subproblem(intervening=ConLin(), approximation=Taylor2(force_convex=False),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x0, prob.g(prob.x0), prob.dg(prob.x0), prob.ddg(prob.x0))
    inter = ConLin()
    inter.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)
    ddfddy = prob.ddg(prob.x0) * (inter.dxdy(prob.x0)) ** 2 + prob.dg(prob.x0) * (inter.ddxddy(prob.x0))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert subprob.dg(prob.x0) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.ddg(prob.x0) == pytest.approx(prob.ddg(prob.x0), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.approx.ddfddy == pytest.approx(ddfddy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x0 + h) - inter.y(prob.x0)).T
    assert subprob.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + np.diag(dfdy.dot(delta_y)) +
                                                  0.5 * np.diag(ddfddy.dot(delta_y ** 2)), rel=1e-4)
    assert subprob.dg(prob.x0 + h) == pytest.approx(dfdy * inter.dydx(prob.x0 + h) +
                                                   ddfddy * delta_y.T * inter.dydx(prob.x0 + h), rel=1e-4)
    assert subprob.ddg(prob.x0 + h) == pytest.approx(dfdy * inter.ddyddx(prob.x0 + h) + ddfddy * delta_y.T *
                                                    inter.ddyddx(prob.x0 + h) + ddfddy * inter.dydx(prob.x0 + h) ** 2,
                                                    rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_mma_taylor1(n, h):
    logger.info("Testing Subproblem(Taylor1, y=MMA)")
    prob = Square(n)
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    inter = MMA(prob.xmin, prob.xmax)
    inter.update(prob.x0, prob.dg(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert subprob.dg(prob.x0) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x0 + h) - inter.y(prob.x0)).T
    assert subprob.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + np.diag(dfdy.dot(delta_y)), rel=1e-4)
    assert subprob.dg(prob.x0 + h) == pytest.approx(dfdy * inter.dydx(prob.x0 + h), rel=1e-4)
    assert subprob.ddg(prob.x0 + h) == pytest.approx(dfdy * inter.ddyddx(prob.x0 + h), rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_mma_taylor2(n, h):
    logger.info("Testing Subproblem(Taylor2, y=MMA)")
    prob = Square(n)
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor2(force_convex=False),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x0, prob.g(prob.x0), prob.dg(prob.x0), prob.ddg(prob.x0))
    inter = MMA(prob.xmin, prob.xmax)
    inter.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    dfdy = prob.dg(prob.x0) * inter.dxdy(prob.x0)
    ddfddy = prob.ddg(prob.x0) * (inter.dxdy(prob.x0)) ** 2 + prob.dg(prob.x0) * (inter.ddxddy(prob.x0))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert subprob.dg(prob.x0) == pytest.approx(prob.dg(prob.x0), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.ddg(prob.x0) == pytest.approx(prob.ddg(prob.x0), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.approx.ddfddy == pytest.approx(ddfddy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x0 + h) - inter.y(prob.x0)).T
    assert subprob.g(prob.x0 + h) == pytest.approx(prob.g(prob.x0) + np.diag(dfdy.dot(delta_y)) +
                                                  0.5 * np.diag(ddfddy.dot(delta_y ** 2)), rel=1e-4)
    assert subprob.dg(prob.x0 + h) == pytest.approx(dfdy * inter.dydx(prob.x0 + h) +
                                                   ddfddy * delta_y.T * inter.dydx(prob.x0 + h), rel=1e-4)
    assert subprob.ddg(prob.x0 + h) == pytest.approx(dfdy * inter.ddyddx(prob.x0 + h) + ddfddy * delta_y.T *
                                                    inter.ddyddx(prob.x0 + h) + ddfddy * inter.dydx(prob.x0 + h) ** 2,
                                                    rel=1e-4)


if __name__ == "__main__":
    test_lin_taylor1(4, 0.1)
    test_lin_taylor2(4, 0.1)
    test_rec_taylor1(4, 0.1)
    test_rec_taylor2(4, 0.1)
    test_conlin_taylor1(4, 0.1)
    test_conlin_taylor2(4, 0.1)
    test_mma_taylor1(4, 0.1)
    test_mma_taylor2(4, 0.1)
