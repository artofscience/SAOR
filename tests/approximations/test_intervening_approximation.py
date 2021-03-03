import pytest
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.approximations.intervening import Linear, Reciprocal, ConLin, MMA
from sao.approximations.bounds import Bounds
from sao.approximations.interveningapproximation import InterveningApproximation


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_lin_taylor1(n):
    print("Testing 1st-order Taylor wrt y=x")
    prob = Square(n)
    approx = InterveningApproximation(intervening=Linear(), approximation=Taylor1(), bounds=Bounds(prob.xmin, prob.xmax))
    approx.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    assert approx.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert approx.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_lin_taylor2(n):
    print("Testing 2nd-order Taylor wrt y=x")
    prob = Square(n)
    approx = InterveningApproximation(intervening=Linear(), approximation=Taylor2(),
                                      bounds=Bounds(prob.xmin, prob.xmax))
    approx.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    assert approx.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert approx.dg(prob.x) == pytest.approx(prob.dg(prob.x), rel=1e-4)
    assert approx.ddg(prob.x) == pytest.approx(prob.ddg(prob.x), rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_rec_taylor1(n):
    print("Testing 1st-order Taylor wrt y=1/x")
    prob = Square(n)
    approx = InterveningApproximation(intervening=Reciprocal(), approximation=Taylor1(),
                                      bounds=Bounds(prob.xmin, prob.xmax))
    approx.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    P = prob.dg(prob.x) * (-(prob.x**2))

    assert approx.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert approx.dg(prob.x) == pytest.approx(P, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_rec_taylor2(n):
    print("Testing 2nd-order Taylor wrt y=1/x")
    prob = Square(n)
    approx = InterveningApproximation(intervening=Reciprocal(), approximation=Taylor2(),
                                      bounds=Bounds(prob.xmin, prob.xmax))
    approx.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    P = prob.dg(prob.x) * (-(prob.x**2))
    Q = prob.ddg(prob.x) * prob.x**4 + prob.dg(prob.x) * 2 * prob.x**3
    Q[Q < 0] = 0        # preserve convexity

    assert approx.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert approx.dg(prob.x) == pytest.approx(P, rel=1e-4)
    assert approx.ddg(prob.x) == pytest.approx(Q, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_conlin_taylor1(n):
    print("Testing 1st-order Taylor with y=ConLin")
    prob = Square(n)
    approx = InterveningApproximation(intervening=ConLin(), approximation=Taylor1(),
                                      bounds=Bounds(prob.xmin, prob.xmax))
    approx.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    conlin = ConLin()
    conlin.update_intervening(prob.dg(prob.x))
    P = prob.dg(prob.x) * conlin.dxdy(prob.x)

    assert approx.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert approx.dg(prob.x) == pytest.approx(P, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_conlin_taylor2(n):
    print("Testing 2nd-order Taylor with y=ConLin")
    prob = Square(n)
    approx = InterveningApproximation(intervening=ConLin(), approximation=Taylor2(),
                                      bounds=Bounds(prob.xmin, prob.xmax))
    approx.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    conlin = ConLin()
    conlin.update_intervening(prob.dg(prob.x))
    P = prob.dg(prob.x) * conlin.dxdy(prob.x)
    Q = prob.ddg(prob.x)*(conlin.dxdy(prob.x))**2 + prob.dg(prob.x)*conlin.ddxddy(prob.x)
    Q[Q < 0] = 0            # preserve convexity

    assert approx.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert approx.dg(prob.x) == pytest.approx(P, rel=1e-4)
    assert approx.ddg(prob.x) == pytest.approx(Q, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_mma_taylor1(n):
    print("Testing 1st-order Taylor with y=MMA")
    prob = Square(n)
    approx = InterveningApproximation(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
                                      bounds=Bounds(prob.xmin, prob.xmax))
    approx.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    mma = MMA(prob.xmin, prob.xmax)
    mma.update_intervening(prob.x, prob.g(prob.x), prob.dg(prob.x))
    P = prob.dg(prob.x) * mma.dxdy(prob.x)

    assert approx.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert approx.dg(prob.x) == pytest.approx(P, rel=1e-4)


@pytest.mark.parametrize('n', [10, 100, 1000])
def test_mma_taylor2(n):
    print("Testing 2nd-order Taylor with y=MMA")
    prob = Square(n)
    approx = InterveningApproximation(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor2(),
                                      bounds=Bounds(prob.xmin, prob.xmax))
    approx.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    mma = MMA(prob.xmin, prob.xmax)
    mma.update_intervening(prob.x, prob.g(prob.x), prob.dg(prob.x))
    P = prob.dg(prob.x) * mma.dxdy(prob.x)
    Q = prob.ddg(prob.x)*(mma.dxdy(prob.x))**2 + prob.dg(prob.x)*mma.ddxddy(prob.x)
    Q[Q < 0] = 0            # preserve convexity

    assert approx.g(prob.x) == pytest.approx(prob.g(prob.x), rel=1e-4)
    assert approx.dg(prob.x) == pytest.approx(P, rel=1e-4)
    assert approx.ddg(prob.x) == pytest.approx(Q, rel=1e-4)


if __name__ == "__main__":
    test_lin_taylor1(4)
    test_lin_taylor2(4)
    test_rec_taylor1(4)
    test_rec_taylor2(4)
    test_conlin_taylor1(4)
    test_conlin_taylor2(4)
    test_mma_taylor1(4)
    test_mma_taylor2(4)


