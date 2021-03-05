import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_taylor1(n):

    print("Testing 1st-order Taylor expansion")
    prob = Square(n)
    taylor1 = Taylor1()
    taylor1.update(prob.x, prob.g(prob.x), prob.dg(prob.x))

    assert taylor1.g(y=prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert taylor1.dg(dy=np.ones_like(prob.x)) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert taylor1.ddg(ddy=np.zeros_like(prob.x)) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_taylor2(n):
    print("Testing 2nd-order Taylor expansion")
    prob = Square(n)
    taylor2 = Taylor2()
    taylor2.update(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    assert taylor2.g(y=prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert taylor2.dg(y=prob.x, dy=np.ones_like(prob.x)) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert taylor2.ddg(y=prob.x, dy=np.ones_like(prob.x), ddy=np.zeros_like(prob.x)) == pytest.approx(prob.ddg(prob.x), rel=1e-4)


if __name__ == "__main__":
    test_taylor1(4)
    test_taylor2(4)

