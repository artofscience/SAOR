import pytest
import logging
import numpy as np
from Problems.square import Square
from sao.intervening_vars.intervening import Linear, Reciprocal, ConLin, MMA

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@pytest.mark.parametrize('n', [10])
def test_linear(n):
    logger.info("Testing linear intervening variables")
    prob = Square(n)
    lin = Linear()

    assert lin.y(prob.x0) == pytest.approx(prob.x0, rel=1e-4)
    assert lin.dydx(prob.x0) == pytest.approx(1, rel=1e-4)
    assert lin.ddyddx(prob.x0) == pytest.approx(0, rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_reciprocal(n):
    logger.info("Testing reciprocal intervening variables")
    prob = Square(n)
    rec = Reciprocal()

    assert rec.y(prob.x0) == pytest.approx(1/prob.x0, rel=1e-4)
    assert rec.dydx(prob.x0) == pytest.approx(-1/prob.x0**2, rel=1e-4)
    assert rec.ddyddx(prob.x0) == pytest.approx(2/prob.x0**3, rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_conlin(n):
    logger.info("Testing ConLin intervening variables")
    prob = Square(n)
    conlin = ConLin()
    conlin.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    temp_y = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_dy = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_ddy = np.zeros((prob.m + 1, prob.n), dtype=float)
    for j in range(0, prob.m + 1):
        temp_y[j, (prob.dg(prob.x0) > 0)[j, :]] = prob.x0[conlin.positive[j, :]]
        temp_y[j, (prob.dg(prob.x0) < 0)[j, :]] = 1 / prob.x0[~conlin.positive[j, :]]
        temp_dy[j, (prob.dg(prob.x0) > 0)[j, :]] = np.ones_like(prob.x0)[conlin.positive[j, :]]
        temp_dy[j, (prob.dg(prob.x0) < 0)[j, :]] = -1 / (prob.x0[~conlin.positive[j, :]])**2
        temp_ddy[j, (prob.dg(prob.x0) > 0)[j, :]] = np.zeros_like(prob.x0)[conlin.positive[j, :]]
        temp_ddy[j, (prob.dg(prob.x0) < 0)[j, :]] = 2 / (prob.x0[~conlin.positive[j, :]])**3

    assert conlin.y(prob.x0) == pytest.approx(temp_y, rel=1e-4)
    assert conlin.dydx(prob.x0) == pytest.approx(temp_dy, rel=1e-4)
    assert conlin.ddyddx(prob.x0) == pytest.approx(temp_ddy, rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_mma(n):
    logger.info("Testing MMA intervening variables")
    prob = Square(n)
    mma = MMA(prob.xmin, prob.xmax)
    mma.update(prob.x0, prob.g(prob.x0), prob.dg(prob.x0))
    temp_y = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_dy = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_ddy = np.zeros((prob.m + 1, prob.n), dtype=float)
    for j in range(0, prob.m + 1):
        temp_y[j, (prob.dg(prob.x0) > 0)[j, :]] = (1 / (mma.upp - prob.x0))[mma.positive[j, :]]
        temp_y[j, (prob.dg(prob.x0) < 0)[j, :]] = (1 / (prob.x0 - mma.low))[~mma.positive[j, :]]
        temp_dy[j, (prob.dg(prob.x0) > 0)[j, :]] = (1 / (mma.upp - prob.x0)**2)[mma.positive[j, :]]
        temp_dy[j, (prob.dg(prob.x0) < 0)[j, :]] = (-1 / (prob.x0 - mma.low)**2)[~mma.positive[j, :]]
        temp_ddy[j, (prob.dg(prob.x0) > 0)[j, :]] = (2 / (mma.upp - prob.x0)**3)[mma.positive[j, :]]
        temp_ddy[j, (prob.dg(prob.x0) < 0)[j, :]] = (2 / (prob.x0 - mma.low)**3)[~mma.positive[j, :]]

    assert mma.y(prob.x0) == pytest.approx(temp_y, rel=1e-4)
    assert mma.dydx(prob.x0) == pytest.approx(temp_dy, rel=1e-4)
    assert mma.ddyddx(prob.x0) == pytest.approx(temp_ddy, rel=1e-4)


if __name__ == "__main__":
    test_linear(4)
    test_reciprocal(4)
    test_conlin(4)
    test_mma(4)
