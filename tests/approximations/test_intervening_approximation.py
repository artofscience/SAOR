import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.approximations.intervening import Linear, Reciprocal, ConLin
from sao.approximations.bounds import Bounds
from sao.approximations.interveningapproximation import InterveningApproximation


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_lin_taylor1(n):
    print("Testing 1st-order Taylor wrt y=x")
    prob = Square(n)
    approx = InterveningApproximation(intervening=Linear(), approximation=Taylor1(), bounds=Bounds(prob.xmin, prob.xmax))

    assert approx.g_approx(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert lin.dy(prob.x) == pytest.approx(1, rel=1e-4)
    assert lin.ddy(prob.x) == pytest.approx(0, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_lin_taylor2(n):
    print("Testing 2nd-order Taylor wrt y=x")
    prob = Square(n)
    lin = Linear()

    assert lin.y(prob.x) == pytest.approx(prob.x, rel=1e-4)
    assert lin.dy(prob.x) == pytest.approx(1, rel=1e-4)
    assert lin.ddy(prob.x) == pytest.approx(0, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_rec_taylor1(n):
    print("Testing 1st-order Taylor wrt y=1/x")
    prob = Square(n)
    lin = Linear()

    assert lin.y(prob.x) == pytest.approx(prob.x, rel=1e-4)
    assert lin.dy(prob.x) == pytest.approx(1, rel=1e-4)
    assert lin.ddy(prob.x) == pytest.approx(0, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_rec_taylor2(n):
    print("Testing 2nd-order Taylor wrt y=1/x")
    prob = Square(n)
    lin = Linear()

    assert lin.y(prob.x) == pytest.approx(prob.x, rel=1e-4)
    assert lin.dy(prob.x) == pytest.approx(1, rel=1e-4)
    assert lin.ddy(prob.x) == pytest.approx(0, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_conlin_taylor1(n):
    print("Testing 1st-order Taylor with ConLin")
    prob = Square(n)
    lin = Linear()

    assert lin.y(prob.x) == pytest.approx(prob.x, rel=1e-4)
    assert lin.dy(prob.x) == pytest.approx(1, rel=1e-4)
    assert lin.ddy(prob.x) == pytest.approx(0, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_conlin_taylor2(n):
    print("Testing 2nd-order Taylor with ConLin")
    prob = Square(n)
    lin = Linear()

    assert lin.y(prob.x) == pytest.approx(prob.x, rel=1e-4)
    assert lin.dy(prob.x) == pytest.approx(1, rel=1e-4)
    assert lin.ddy(prob.x) == pytest.approx(0, rel=1e-4)


if __name__ == "__main__":
    test_lin_taylor1(4)
    test_lin_taylor2(4)
    test_rec_taylor1(4)
    test_rec_taylor2(4)
    test_conlin_taylor1(4)
    test_conlin_taylor2(4)
