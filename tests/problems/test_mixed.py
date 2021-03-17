import pytest
import numpy as np
import logging
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.approximations.intervening import Linear, Reciprocal, ConLin, MMA
from sao.move_limits.ml_intervening import MoveLimitIntervening
from sao.problems.subproblem import Subproblem
from sao.problems.mixed import Mixed

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
def test_mixed(n, h):

    ## Initializations
    logger.info("Testing Subproblem(Taylor1, y=x)")
    prob = Square(n)

    # Define variable and response sets of a mixed approximation scheme as dictionaries
    var_set = {0: np.array([0, 1]),
               1: np.arange(2, prob.n)}
    resp_set = {0: np.array([0]),
                1: np.array([1])}

    # Instantiate subproblem objects for a mixed approximation scheme
    subprob_map = {(0, 0): Subproblem(intervening=Linear(),
                                      approximation=Taylor2(force_convex=False),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]])),
                   (0, 1): Subproblem(intervening=Reciprocal(),
                                      approximation=Taylor2(force_convex=False),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[1]],
                                                              xmax=prob.xmax[var_set[1]])),
                   (1, 0): Subproblem(intervening=Reciprocal(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]])),
                   (1, 1): Subproblem(intervening=Linear(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[1]],
                                                              xmax=prob.xmax[var_set[1]]))}

    # Instantiate a mixed scheme
    subprob = Mixed(subprob_map, var_set, resp_set)

    # Build manually the arrays for reciprocal intervening variables
    dfdy_rec = prob.dg(prob.x0) * (-(prob.x0 ** 2))
    ddfddy_rec = prob.ddg(prob.x0) * prob.x0 ** 4 + prob.dg(prob.x0) * 2 * prob.x0 ** 3
    delta_y = (subprob_map[0, 1].inter.y(prob.x0 + h) - subprob_map[0, 1].inter.y(prob.x0)).T

    # Build the subproblem resulting from a mixed scheme at the current point
    subprob.build(prob.x0, prob.g(prob.x0), prob.dg(prob.x0), prob.ddg(prob.x0))

    ## Tests at expansion point X^(k)

    # Check validity of approximate responses (and sensitivities) at expansion point X^(k)
    assert subprob.g(prob.x0) == pytest.approx(prob.g(prob.x0), rel=1e-4)
    assert subprob.dg(prob.x0) == pytest.approx(prob.dg(prob.x0), rel=1e-4)

    # Check 2nd-order sensitivities only for the 2nd-order Taylor expansions, i.e. [resp_set[0], :], at X^(k)
    assert subprob.ddg(prob.x0)[resp_set[0], :] == pytest.approx(prob.ddg(prob.x0)[resp_set[0], :], rel=1e-4)

    ## Tests for g(X^(k) + h)

    # Check g_0(X^(k) + h)
    assert subprob.g(prob.x0 + h)[resp_set[0]] == pytest.approx(
        prob.g(prob.x0)[resp_set[0]] +
        np.dot(prob.dg(prob.x0)[np.ix_(resp_set[0], var_set[0])], h * np.ones_like(prob.x0)[var_set[0]]) +
        0.5 * np.dot(prob.ddg(prob.x0)[np.ix_(resp_set[0], var_set[0])], h ** 2 * np.ones_like(prob.x0)[var_set[0]]) +
        np.dot(dfdy_rec[np.ix_(resp_set[0], var_set[1])], delta_y[var_set[1]]) +
        0.5 * np.dot(ddfddy_rec[np.ix_(resp_set[0], var_set[1])], delta_y[var_set[1]]**2), rel=1e-4)

    # Check g_1(X^(k) + h)
    assert subprob.g(prob.x0 + h)[resp_set[1]] == pytest.approx(
        prob.g(prob.x0)[resp_set[1]] +
        np.dot(prob.dg(prob.x0)[np.ix_(resp_set[1], var_set[1])], h * np.ones_like(prob.x0)[var_set[1]]) +
        np.dot(dfdy_rec[np.ix_(resp_set[1], var_set[0])], delta_y[var_set[0]]), rel=1e-4)

    ## Tests for dg(X^(k) + h)

    # Check dg(X^(k) + h) for Linear + Taylor2 of subprob_map[0, 0]
    assert subprob.dg(prob.x0 + h)[np.ix_(resp_set[0], var_set[0])] == pytest.approx(
        prob.dg(prob.x0)[np.ix_(resp_set[0], var_set[0])] +
        prob.ddg(prob.x0)[np.ix_(resp_set[0], var_set[0])] * h, rel=1e-4)

    # Check dg(X^(k) + h) for  Reciprocal + Taylor2 of subprob_map[0, 1]
    assert subprob.dg(prob.x0 + h)[np.ix_(resp_set[0], var_set[1])] == pytest.approx(
        dfdy_rec[np.ix_(resp_set[0], var_set[1])] * subprob_map[0, 1].inter.dydx(prob.x0 + h)[var_set[1]] +
        ddfddy_rec[np.ix_(resp_set[0], var_set[1])] * delta_y[var_set[1]].T * subprob_map[0, 1].inter.dydx(prob.x0[var_set[1]] + h), rel=1e-4)

    # Check dg(X^(k) + h) for Reciprocal + Taylor1 of subprob_map[1, 0]
    assert subprob.dg(prob.x0 + h)[np.ix_(resp_set[1], var_set[0])] == pytest.approx(
        dfdy_rec[np.ix_(resp_set[1], var_set[0])] * subprob_map[1, 0].inter.dydx(prob.x0[var_set[0]] + h), rel=1e-4)

    # Check dg(X^(k) + h) for Linear + Taylor1 of subprob_map[1, 1]
    assert subprob.dg(prob.x0 + h)[np.ix_(resp_set[1], var_set[1])] == pytest.approx(
        prob.dg(prob.x0)[np.ix_(resp_set[1], var_set[1])], rel=1e-4)

    ## Tests for ddg(X^(k) + h)

    # Check ddg(X^(k) + h) for Linear + Taylor2 of subprob_map[0, 0]
    assert subprob.ddg(prob.x0 + h)[np.ix_(resp_set[0], var_set[0])] == pytest.approx(
        prob.ddg(prob.x0)[np.ix_(resp_set[0], var_set[0])] * np.ones_like(prob.x0)[var_set[0]]**2, rel=1e-4)

    # Check ddg(X^(k) + h) for  Reciprocal + Taylor2 of subprob_map[0, 1]
    assert subprob.ddg(prob.x0 + h)[np.ix_(resp_set[0], var_set[1])] == pytest.approx(
        dfdy_rec[np.ix_(resp_set[0], var_set[1])] * subprob_map[0, 1].inter.ddyddx(prob.x0[var_set[1]] + h) +
        ddfddy_rec[np.ix_(resp_set[0], var_set[1])] * delta_y[var_set[1]].T * subprob_map[0, 1].inter.ddyddx(prob.x0[var_set[1]] + h) +
        ddfddy_rec[np.ix_(resp_set[0], var_set[1])] * subprob_map[0, 1].inter.dydx(prob.x0[var_set[1]] + h) ** 2, rel=1e-4)

    # Check ddg(X^(k) + h) for Reciprocal + Taylor1 of subprob_map[1, 0]
    assert subprob.ddg(prob.x0 + h)[np.ix_(resp_set[1], var_set[0])] == pytest.approx(
        dfdy_rec[np.ix_(resp_set[1], var_set[0])] * subprob_map[1, 0].inter.ddyddx(prob.x0[var_set[0]] + h), rel=1e-4)

    # Check ddg(X^(k) + h) for Linear + Taylor1 of subprob_map[1, 1]
    assert subprob.ddg(prob.x0 + h)[np.ix_(resp_set[1], var_set[1])] == pytest.approx(0, abs=1e-4)


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


if __name__ == "__main__":
    test_mixed(4, 0.1)

