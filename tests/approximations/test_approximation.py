import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.taylor import Taylor1
from sao.approximations.taylor import Taylor2


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_taylor1(n):

    print("Test 1st-order Taylor expansion")
    prob = Square(n)
    taylor1 = Taylor1()
    taylor1.update_approximation(prob.x, prob.xmin, prob.xmax, prob.g(prob.x), prob.dg(prob.x))

    assert taylor1.g_approx(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert taylor1.dg_approx(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert taylor1.ddg_approx(prob.x) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_taylor2(n):
    print("Test 1st-order Taylor expansion")
    prob = Square(n)
    taylor2 = Taylor2()
    taylor2.update_approximation(prob.x, prob.xmin, prob.xmax, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    assert taylor2.g_approx(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert taylor2.dg_approx(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert taylor2.ddg_approx(prob.x) == pytest.approx(prob.ddg(prob.x), rel=1e-4)


if __name__ == "__main__":
    test_taylor1(4)
    test_taylor2(4)

