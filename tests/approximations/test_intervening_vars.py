import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.intervening import Linear, Reciprocal, ConLin


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_linear(n):
    print("Testing linear intervening variables")
    prob = Square(n)
    lin = Linear()

    assert lin.y(prob.x) == pytest.approx(prob.x, rel=1e-4)
    assert lin.dy(prob.x) == pytest.approx(1, rel=1e-4)
    assert lin.ddy(prob.x) == pytest.approx(0, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_reciprocal(n):
    print("Testing linear intervening variables")
    prob = Square(n)
    rec = Reciprocal()

    assert rec.y(prob.x) == pytest.approx(1/prob.x, rel=1e-4)
    assert rec.dy(prob.x) == pytest.approx(-1/prob.x**2, rel=1e-4)
    assert rec.ddy(prob.x) == pytest.approx(2/prob.x**3, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_conlin(n):
    print("Testing linear intervening variables")
    prob = Square(n)
    conlin = ConLin()
    conlin.update_intervening(prob.dg(prob.x))
    temp_y = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_dy = np.zeros((prob.m + 1, prob.n), dtype=float)
    temp_ddy = np.zeros((prob.m + 1, prob.n), dtype=float)
    for j in range(0, prob.m + 1):
        temp_y[j, conlin.positive[j, :]] = prob.x[conlin.positive[j, :]]
        temp_y[j, conlin.negative[j, :]] = 1 / prob.x[conlin.negative[j, :]]
        temp_dy[j, conlin.positive[j, :]] = np.ones_like(prob.x)[conlin.positive[j, :]]
        temp_dy[j, conlin.negative[j, :]] = -1 / (prob.x[conlin.negative[j, :]])**2
        temp_ddy[j, conlin.positive[j, :]] = np.zeros_like(prob.x)[conlin.positive[j, :]]
        temp_ddy[j, conlin.negative[j, :]] = 2 / (prob.x[conlin.negative[j, :]])**3

    assert conlin.y(prob.x) == pytest.approx(temp_y, rel=1e-4)
    assert conlin.dy(prob.x) == pytest.approx(temp_dy, rel=1e-4)
    assert conlin.ddy(prob.x) == pytest.approx(temp_ddy, rel=1e-4)


if __name__ == "__main__":
    test_linear(4)
    test_reciprocal(4)
    test_conlin(4)


