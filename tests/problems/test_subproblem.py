import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.approximations.intervening import Linear, Reciprocal, ConLin, MMA
from sao.move_limits.ml_intervening import MoveLimitIntervening
from sao.problems.subproblem import Subproblem


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_lin_taylor1(n, h):
    print("Testing Subproblem(Taylor1, y=x)")
    prob = Square(n)
    subprob = Subproblem(intervening=Linear(), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x, prob.g(prob.x), prob.dg(prob.x))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert subprob.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    assert subprob.g(prob.x + h) == pytest.approx(prob.g(prob.x) + np.dot(prob.dg(prob.x), h*np.ones_like(prob.x)), rel=1e-4)
    assert subprob.dg(prob.x + h) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert subprob.ddg(prob.x + h) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_lin_taylor2(n, h):
    print("Subproblem(Taylor2, y=x)")
    prob = Square(n)
    subprob = Subproblem(intervening=Linear(), approximation=Taylor2(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert subprob.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert subprob.ddg(prob.x) == pytest.approx(prob.ddg(prob.x), rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    assert subprob.g(prob.x + h) == pytest.approx(prob.g(prob.x) + np.dot(prob.dg(prob.x), h*np.ones_like(prob.x)) +
                                                  0.5 * np.dot(prob.ddg(prob.x), h**2*np.ones_like(prob.x)), rel=1e-4)
    assert subprob.dg(prob.x + h) == pytest.approx(prob.dg(prob.x) + prob.ddg(prob.x) * h, rel=1e-4)
    assert subprob.ddg(prob.x + h) == pytest.approx(prob.ddg(prob.x) * np.ones_like(prob.x)**2, rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_rec_taylor1(n, h):
    print("Testing Subproblem(Taylor1, y=1/x)")
    prob = Square(n)
    subprob = Subproblem(intervening=Reciprocal(), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x, prob.g(prob.x), prob.dg(prob.x))
    inter = Reciprocal()
    inter.update(prob.x, prob.dg(prob.x), prob.dg(prob.x))
    dfdy = prob.dg(prob.x) * (-(prob.x**2))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert subprob.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert dfdy == pytest.approx(prob.dg(prob.x) * inter.dxdy(prob.x), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x + h) - inter.y(prob.x)).T
    assert subprob.g(prob.x + h) == pytest.approx(prob.g(prob.x) + dfdy.dot(delta_y), rel=1e-4)
    assert subprob.dg(prob.x + h) == pytest.approx(dfdy * inter.dydx(prob.x + h), rel=1e-4)
    assert subprob.ddg(prob.x + h) == pytest.approx(dfdy * inter.ddyddx(prob.x + h), rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_rec_taylor2(n, h):
    print("Testing Subproblem(Taylor2, y=1/x)")
    prob = Square(n)
    subprob = Subproblem(intervening=Reciprocal(), approximation=Taylor2(force_convex=False),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))
    inter = Reciprocal()
    inter.update(prob.x, prob.g(prob.x), prob.dg(prob.x))
    dfdy = prob.dg(prob.x) * (-(prob.x**2))
    ddfddy = prob.ddg(prob.x) * prob.x**4 + prob.dg(prob.x) * 2 * prob.x**3

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert subprob.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.ddg(prob.x) == pytest.approx(prob.ddg(prob.x), rel=1e-4)
    assert dfdy == pytest.approx(prob.dg(prob.x) * inter.dxdy(prob.x), rel=1e-4)
    assert ddfddy == pytest.approx(prob.ddg(prob.x) * (inter.dxdy(prob.x)) ** 2 + prob.dg(prob.x) * (inter.ddxddy(prob.x)), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.approx.ddfddy == pytest.approx(ddfddy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x + h) - inter.y(prob.x)).T
    assert subprob.g(prob.x + h) == pytest.approx(prob.g(prob.x) + dfdy.dot(delta_y) + 0.5 * ddfddy.dot(delta_y ** 2), rel=1e-4)
    assert subprob.dg(prob.x + h) == pytest.approx(dfdy * inter.dydx(prob.x + h) + ddfddy * delta_y.T * inter.dydx(prob.x + h), rel=1e-4)
    assert subprob.ddg(prob.x + h) == pytest.approx(dfdy * inter.ddyddx(prob.x + h) + ddfddy * delta_y.T *
                                                    inter.ddyddx(prob.x + h) + ddfddy * inter.dydx(prob.x + h) ** 2, rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_conlin_taylor1(n, h):
    print("Testing Subproblem(Taylor1, y=ConLin)")
    prob = Square(n)
    subprob = Subproblem(intervening=ConLin(), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x, prob.g(prob.x), prob.dg(prob.x))
    inter = ConLin()
    inter.update(prob.x, prob.dg(prob.x), prob.dg(prob.x))
    dfdy = prob.dg(prob.x) * inter.dxdy(prob.x)

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert subprob.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x + h) - inter.y(prob.x)).T
    assert subprob.g(prob.x + h) == pytest.approx(prob.g(prob.x) + np.diag(dfdy.dot(delta_y)), rel=1e-4)
    assert subprob.dg(prob.x + h) == pytest.approx(dfdy * inter.dydx(prob.x + h), rel=1e-4)
    assert subprob.ddg(prob.x + h) == pytest.approx(dfdy * inter.ddyddx(prob.x + h), rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_conlin_taylor2(n, h):
    print("Testing Subproblem(Taylor2, y=ConLin)")
    prob = Square(n)
    subprob = Subproblem(intervening=ConLin(), approximation=Taylor2(force_convex=False),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))
    inter = ConLin()
    inter.update(prob.x, prob.g(prob.x), prob.dg(prob.x))
    dfdy = prob.dg(prob.x) * inter.dxdy(prob.x)
    ddfddy = prob.ddg(prob.x) * (inter.dxdy(prob.x)) ** 2 + prob.dg(prob.x) * (inter.ddxddy(prob.x))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert subprob.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.ddg(prob.x) == pytest.approx(prob.ddg(prob.x), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.approx.ddfddy == pytest.approx(ddfddy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x + h) - inter.y(prob.x)).T
    assert subprob.g(prob.x + h) == pytest.approx(prob.g(prob.x) + np.diag(dfdy.dot(delta_y)) +
                                                  0.5 * np.diag(ddfddy.dot(delta_y ** 2)), rel=1e-4)
    assert subprob.dg(prob.x + h) == pytest.approx(dfdy * inter.dydx(prob.x + h) +
                                                   ddfddy * delta_y.T * inter.dydx(prob.x + h), rel=1e-4)
    assert subprob.ddg(prob.x + h) == pytest.approx(dfdy * inter.ddyddx(prob.x + h) + ddfddy * delta_y.T *
                                                    inter.ddyddx(prob.x + h) + ddfddy * inter.dydx(prob.x + h) ** 2,
                                                    rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_mma_taylor1(n, h):
    print("Testing Subproblem(Taylor1, y=MMA)")
    prob = Square(n)
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x, prob.g(prob.x), prob.dg(prob.x))
    inter = MMA(prob.xmin, prob.xmax)
    inter.update(prob.x, prob.dg(prob.x), prob.dg(prob.x))
    dfdy = prob.dg(prob.x) * inter.dxdy(prob.x)

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert subprob.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x + h) - inter.y(prob.x)).T
    assert subprob.g(prob.x + h) == pytest.approx(prob.g(prob.x) + np.diag(dfdy.dot(delta_y)), rel=1e-4)
    assert subprob.dg(prob.x + h) == pytest.approx(dfdy * inter.dydx(prob.x + h), rel=1e-4)
    assert subprob.ddg(prob.x + h) == pytest.approx(dfdy * inter.ddyddx(prob.x + h), rel=1e-4)


@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('h', [0.1, 0.5])
def test_mma_taylor2(n, h):
    print("Testing Subproblem(Taylor2, y=MMA)")
    prob = Square(n)
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor2(force_convex=False),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob.build(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))
    inter = MMA(prob.xmin, prob.xmax)
    inter.update(prob.x, prob.g(prob.x), prob.dg(prob.x))
    dfdy = prob.dg(prob.x) * inter.dxdy(prob.x)
    ddfddy = prob.ddg(prob.x) * (inter.dxdy(prob.x)) ** 2 + prob.dg(prob.x) * (inter.ddxddy(prob.x))

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert subprob.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.ddg(prob.x) == pytest.approx(prob.ddg(prob.x), rel=1e-4)
    assert subprob.approx.dfdy == pytest.approx(dfdy, rel=1e-4)
    assert subprob.approx.ddfddy == pytest.approx(ddfddy, rel=1e-4)

    # Check validity of approximate responses (and sensitivities) at X^(k) + h
    delta_y = (inter.y(prob.x + h) - inter.y(prob.x)).T
    assert subprob.g(prob.x + h) == pytest.approx(prob.g(prob.x) + np.diag(dfdy.dot(delta_y)) +
                                                  0.5 * np.diag(ddfddy.dot(delta_y ** 2)), rel=1e-4)
    assert subprob.dg(prob.x + h) == pytest.approx(dfdy * inter.dydx(prob.x + h) +
                                                   ddfddy * delta_y.T * inter.dydx(prob.x + h), rel=1e-4)
    assert subprob.ddg(prob.x + h) == pytest.approx(dfdy * inter.ddyddx(prob.x + h) + ddfddy * delta_y.T *
                                                    inter.ddyddx(prob.x + h) + ddfddy * inter.dydx(prob.x + h) ** 2,
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
