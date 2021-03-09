import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.intervening import Linear, Reciprocal, ConLin, MMA


@pytest.mark.parametrize('n', [10])
def test_linear(n):
    print("Testing linear intervening variables")
    prob = Square(n)
    lin = Linear()

    assert lin.y(prob.x) == pytest.approx(prob.x, rel=1e-4)
    assert lin.dydx(prob.x) == pytest.approx(1, rel=1e-4)
    assert lin.ddyddx(prob.x) == pytest.approx(0, rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_reciprocal(n):
    print("Testing reciprocal intervening variables")
    prob = Square(n)
    rec = Reciprocal()

    assert rec.y(prob.x) == pytest.approx(1/prob.x, rel=1e-4)
    assert rec.dydx(prob.x) == pytest.approx(-1/prob.x**2, rel=1e-4)
    assert rec.ddyddx(prob.x) == pytest.approx(2/prob.x**3, rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_conlin(n):
    print("Testing ConLin intervening variables")
    prob = Square(n)
    conlin = ConLin()
    conlin.update(prob.x, prob.g(prob.x), prob.dg(prob.x))
    temp_y = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_dy = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_ddy = np.zeros((prob.m + 1, prob.n), dtype=float)
    for j in range(0, prob.m + 1):
        temp_y[j, (prob.dg(prob.x) > 0)[j, :]] = prob.x[conlin.positive[j, :]]
        temp_y[j, (prob.dg(prob.x) < 0)[j, :]] = 1 / prob.x[~conlin.positive[j, :]]
        temp_dy[j, (prob.dg(prob.x) > 0)[j, :]] = np.ones_like(prob.x)[conlin.positive[j, :]]
        temp_dy[j, (prob.dg(prob.x) < 0)[j, :]] = -1 / (prob.x[~conlin.positive[j, :]])**2
        temp_ddy[j, (prob.dg(prob.x) > 0)[j, :]] = np.zeros_like(prob.x)[conlin.positive[j, :]]
        temp_ddy[j, (prob.dg(prob.x) < 0)[j, :]] = 2 / (prob.x[~conlin.positive[j, :]])**3

    assert conlin.y(prob.x) == pytest.approx(temp_y, rel=1e-4)
    assert conlin.dydx(prob.x) == pytest.approx(temp_dy, rel=1e-4)
    assert conlin.ddyddx(prob.x) == pytest.approx(temp_ddy, rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_mma(n):
    print("Testing MMA intervening variables")
    prob = Square(n)
    mma = MMA(prob.xmin, prob.xmax)
    mma.update(prob.x, prob.g(prob.x), prob.dg(prob.x))
    temp_y = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_dy = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_ddy = np.zeros((prob.m + 1, prob.n), dtype=float)
    for j in range(0, prob.m + 1):
        temp_y[j, (prob.dg(prob.x) > 0)[j, :]] = (1 / (mma.upp - prob.x))[mma.positive[j, :]]
        temp_y[j, (prob.dg(prob.x) < 0)[j, :]] = (1 / (prob.x - mma.low))[~mma.positive[j, :]]
        temp_dy[j, (prob.dg(prob.x) > 0)[j, :]] = (1 / (mma.upp - prob.x)**2)[mma.positive[j, :]]
        temp_dy[j, (prob.dg(prob.x) < 0)[j, :]] = (-1 / (prob.x - mma.low)**2)[~mma.positive[j, :]]
        temp_ddy[j, (prob.dg(prob.x) > 0)[j, :]] = (2 / (mma.upp - prob.x)**3)[mma.positive[j, :]]
        temp_ddy[j, (prob.dg(prob.x) < 0)[j, :]] = (2 / (prob.x - mma.low)**3)[~mma.positive[j, :]]

    assert mma.y(prob.x) == pytest.approx(temp_y, rel=1e-4)
    assert mma.dydx(prob.x) == pytest.approx(temp_dy, rel=1e-4)
    assert mma.ddyddx(prob.x) == pytest.approx(temp_ddy, rel=1e-4)


if __name__ == "__main__":
    test_linear(4)
    test_reciprocal(4)
    test_conlin(4)
    test_mma(4)