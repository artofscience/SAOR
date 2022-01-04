import pytest
import logging
import numpy as np
from Problems._nd.Square import Square
from sao.intervening_variables import Linear, Reciprocal, MixedIntervening, Exponential
from sao.intervening_variables import MMA02 as MMA
from sao.intervening_variables.mixed_intervening import fill_set_when_emtpy
from sao.intervening_variables.asymptote_update_strategies import Svanberg2002
# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@pytest.mark.parametrize('n, s, out', [
    (5, None, set(range(5))),
    (5, ..., set(range(5))),
    (5, 1, {1}),
    (5, {1, 2}, {1, 2}),
    (5, [1, 2], {1, 2}),
    (5, np.array([1, 2]), {1, 2}),
])
def test_ensure_non_empty_set_with_size(n, s, out):
    assert fill_set_when_emtpy(s, n) == out


@pytest.mark.parametrize('n', [10])
def test_uniform(n):
    logger.info("Testing uniform mixed intervening variables")
    prob = Square(n)

    mix = MixedIntervening(prob.n, prob.m + 1, default=Linear())
    lin = Linear()

    assert lin.y(prob.x0) == pytest.approx(mix.y(prob.x0)[0], rel=1e-4)
    assert lin.y(prob.x0) == pytest.approx(mix.y(prob.x0)[1], rel=1e-4)
    assert lin.dydx(prob.x0) == pytest.approx(mix.dydx(prob.x0)[0], rel=1e-4)
    assert lin.dydx(prob.x0) == pytest.approx(mix.dydx(prob.x0)[1], rel=1e-4)
    assert lin.ddyddx(prob.x0) == pytest.approx(mix.ddyddx(prob.x0)[0], rel=1e-4)
    assert lin.ddyddx(prob.x0) == pytest.approx(mix.ddyddx(prob.x0)[1], rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_different_per_response(n):
    logger.info("Testing mixed response intervening variables (substitution)")
    prob = Square(n)

    mix = MixedIntervening(prob.n, prob.m + 1, default=Linear())
    mix.set_intervening(Reciprocal(), resp=1)
    lin = Linear()
    rec = Reciprocal()

    assert lin.y(prob.x0) == pytest.approx(mix.y(prob.x0)[0], rel=1e-4)
    assert rec.y(prob.x0) == pytest.approx(mix.y(prob.x0)[1], rel=1e-4)
    assert lin.dydx(prob.x0) == pytest.approx(mix.dydx(prob.x0)[0], rel=1e-4)
    assert rec.dydx(prob.x0) == pytest.approx(mix.dydx(prob.x0)[1], rel=1e-4)
    assert lin.ddyddx(prob.x0) == pytest.approx(mix.ddyddx(prob.x0)[0], rel=1e-4)
    assert rec.ddyddx(prob.x0) == pytest.approx(mix.ddyddx(prob.x0)[1], rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_add_per_response(n):
    logger.info("Testing mixed response intervening variables (addition)")
    prob = Square(n)

    mix = MixedIntervening(prob.n, prob.m + 1, default=Linear())
    mix.add_intervening(Reciprocal(), resp=1)
    lin = Linear()
    rec = Reciprocal()

    assert lin.y(prob.x0) == pytest.approx(mix.y(prob.x0)[0], rel=1e-4)
    assert lin.y(prob.x0)+rec.y(prob.x0) == pytest.approx(mix.y(prob.x0)[1], rel=1e-4)
    assert lin.dydx(prob.x0) == pytest.approx(mix.dydx(prob.x0)[0], rel=1e-4)
    assert lin.dydx(prob.x0)+rec.dydx(prob.x0) == pytest.approx(mix.dydx(prob.x0)[1], rel=1e-4)
    assert lin.ddyddx(prob.x0) == pytest.approx(mix.ddyddx(prob.x0)[0], rel=1e-4)
    assert lin.ddyddx(prob.x0)+rec.ddyddx(prob.x0) == pytest.approx(mix.ddyddx(prob.x0)[1], rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_different_per_variable(n):
    logger.info("Testing mixed variable intervening variables (substitution)")
    prob = Square(n)

    mix = MixedIntervening(prob.n, prob.m + 1, default=Linear())
    mix.set_intervening(Reciprocal(), var=[0, 1])
    lin = Linear()
    rec = Reciprocal()

    assert lin.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[0, 2:], rel=1e-4)
    assert lin.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[1, 2:], rel=1e-4)
    assert rec.y(prob.x0)[:2] == pytest.approx(mix.y(prob.x0)[0, :2], rel=1e-4)
    assert rec.y(prob.x0)[:2] == pytest.approx(mix.y(prob.x0)[1, :2], rel=1e-4)
    assert lin.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[0, 2:], rel=1e-4)
    assert lin.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[1, 2:], rel=1e-4)
    assert rec.dydx(prob.x0)[:2] == pytest.approx(mix.dydx(prob.x0)[0, :2], rel=1e-4)
    assert rec.dydx(prob.x0)[:2] == pytest.approx(mix.dydx(prob.x0)[1, :2], rel=1e-4)
    assert lin.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[0, 2:], rel=1e-4)
    assert lin.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[1, 2:], rel=1e-4)
    assert rec.ddyddx(prob.x0)[:2] == pytest.approx(mix.ddyddx(prob.x0)[0, :2], rel=1e-4)
    assert rec.ddyddx(prob.x0)[:2] == pytest.approx(mix.ddyddx(prob.x0)[1, :2], rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_add_per_variable(n):
    logger.info("Testing mixed variable intervening variables (addition)")
    prob = Square(n)

    mix = MixedIntervening(prob.n, prob.m + 1, default=Linear())
    mix.add_intervening(Reciprocal(), var=[0, 1])
    lin = Linear()
    rec = Reciprocal()

    assert lin.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[0, 2:], rel=1e-4)
    assert lin.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[1, 2:], rel=1e-4)
    assert lin.y(prob.x0)[:2] + rec.y(prob.x0)[:2] == pytest.approx(mix.y(prob.x0)[0, :2], rel=1e-4)
    assert lin.y(prob.x0)[:2] + rec.y(prob.x0)[:2] == pytest.approx(mix.y(prob.x0)[1, :2], rel=1e-4)
    assert lin.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[0, 2:], rel=1e-4)
    assert lin.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[1, 2:], rel=1e-4)
    assert lin.dydx(prob.x0)[:2] + rec.dydx(prob.x0)[:2] == pytest.approx(mix.dydx(prob.x0)[0, :2], rel=1e-4)
    assert lin.dydx(prob.x0)[:2] + rec.dydx(prob.x0)[:2] == pytest.approx(mix.dydx(prob.x0)[1, :2], rel=1e-4)
    assert lin.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[0, 2:], rel=1e-4)
    assert lin.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[1, 2:], rel=1e-4)
    assert lin.ddyddx(prob.x0)[:2] + rec.ddyddx(prob.x0)[:2] == pytest.approx(mix.ddyddx(prob.x0)[0, :2], rel=1e-4)
    assert lin.ddyddx(prob.x0)[:2] + rec.ddyddx(prob.x0)[:2] == pytest.approx(mix.ddyddx(prob.x0)[1, :2], rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_different_per_variable_and_response(n):
    logger.info("Testing mixed variable and response intervening variables (substitution)")
    prob = Square(n)

    mix = MixedIntervening(prob.n, prob.m + 1, default=Linear())
    mix.set_intervening(Reciprocal(), var=np.arange(2, n), resp=0)
    mix.set_intervening(Exponential(-2), var=np.arange(0, 2), resp=1)
    mix.set_intervening(Exponential(2), var=np.arange(2, n), resp=1)
    int00 = Linear()
    int01 = Reciprocal()
    int10 = Exponential(-2)
    int11 = Exponential(2)

    assert int00.y(prob.x0)[:2] == pytest.approx(mix.y(prob.x0)[0, :2], rel=1e-4)
    assert int01.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[0, 2:], rel=1e-4)
    assert int10.y(prob.x0)[:2] == pytest.approx(mix.y(prob.x0)[1, :2], rel=1e-4)
    assert int11.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[1, 2:], rel=1e-4)
    assert int00.dydx(prob.x0)[:2] == pytest.approx(mix.dydx(prob.x0)[0, :2], rel=1e-4)
    assert int01.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[0, 2:], rel=1e-4)
    assert int10.dydx(prob.x0)[:2] == pytest.approx(mix.dydx(prob.x0)[1, :2], rel=1e-4)
    assert int11.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[1, 2:], rel=1e-4)
    assert int00.ddyddx(prob.x0)[:2] == pytest.approx(mix.ddyddx(prob.x0)[0, :2], rel=1e-4)
    assert int01.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[0, 2:], rel=1e-4)
    assert int10.ddyddx(prob.x0)[:2] == pytest.approx(mix.ddyddx(prob.x0)[1, :2], rel=1e-4)
    assert int11.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[1, 2:], rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_add_per_variable_and_response(n):
    logger.info("Testing mixed variable and response intervening variables (addition)")
    prob = Square(n)

    mix = MixedIntervening(prob.n, prob.m + 1, default=Linear())
    mix.add_intervening(Reciprocal(), var=np.arange(2, n), resp=0)
    mix.add_intervening(Exponential(-2), var=np.arange(0, 2), resp=1)
    mix.add_intervening(Exponential(2), var=np.arange(2, n), resp=1)
    int00 = Linear()
    int01 = Reciprocal()
    int10 = Exponential(-2)
    int11 = Exponential(2)

    assert int00.y(prob.x0)[:2] == pytest.approx(mix.y(prob.x0)[0, :2], rel=1e-4)
    assert int00.y(prob.x0)[2:]+int01.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[0, 2:], rel=1e-4)
    assert int00.y(prob.x0)[:2]+int10.y(prob.x0)[:2] == pytest.approx(mix.y(prob.x0)[1, :2], rel=1e-4)
    assert int00.y(prob.x0)[2:]+int11.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[1, 2:], rel=1e-4)
    assert int00.dydx(prob.x0)[:2] == pytest.approx(mix.dydx(prob.x0)[0, :2], rel=1e-4)
    assert int00.dydx(prob.x0)[2:]+int01.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[0, 2:], rel=1e-4)
    assert int00.dydx(prob.x0)[:2]+int10.dydx(prob.x0)[:2] == pytest.approx(mix.dydx(prob.x0)[1, :2], rel=1e-4)
    assert int00.dydx(prob.x0)[2:]+int11.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[1, 2:], rel=1e-4)
    assert int00.ddyddx(prob.x0)[:2] == pytest.approx(mix.ddyddx(prob.x0)[0, :2], rel=1e-4)
    assert int00.ddyddx(prob.x0)[2:]+int01.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[0, 2:], rel=1e-4)
    assert int00.ddyddx(prob.x0)[:2]+int10.ddyddx(prob.x0)[:2] == pytest.approx(mix.ddyddx(prob.x0)[1, :2], rel=1e-4)
    assert int00.ddyddx(prob.x0)[2:]+int11.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[1, 2:], rel=1e-4)


@pytest.mark.parametrize('n', [10])
def test_add_per_variable_and_response_multiple_overlap(n):
    logger.info("Testing multiple overlapping mixed variable and response intervening variables (addition)")
    prob = Square(n)

    mix = MixedIntervening(prob.n, prob.m + 1, default=Linear())
    mix.add_intervening(Reciprocal(), resp=[0, 1], var=np.arange(2, n))
    mix.add_intervening(Exponential(-2), resp=1)
    mix.add_intervening(MMA(updaterule=Svanberg2002(x_min=prob.x_min, x_max=prob.x_max)), resp=[0, 1], var=np.arange(2))
    mix.add_intervening(Exponential(3), resp=0, var=np.arange(n))
    intAA = Linear()
    intA1 = Reciprocal()
    int1A = Exponential(-2)
    intA0 = MMA(updaterule=Svanberg2002(x_min=prob.x_min, x_max=prob.x_max))
    int0A = Exponential(3)

    g = prob.g(prob.x0)
    dg = prob.dg(prob.x0)
    ddg = prob.ddg(prob.x0)
    intA0.update(prob.x0, g, dg, ddg)
    mix.update(prob.x0, g, dg, ddg)

    assert intAA.y(prob.x0)[:2]+int0A.y(prob.x0)[:2]+intA0.y(prob.x0)[0, :2] == pytest.approx(mix.y(prob.x0)[0, :2], rel=1e-4)
    assert intAA.y(prob.x0)[2:]+int0A.y(prob.x0)[2:]+intA1.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[0, 2:], rel=1e-4)
    assert intAA.y(prob.x0)[:2]+int1A.y(prob.x0)[:2]+intA0.y(prob.x0)[1, :2] == pytest.approx(mix.y(prob.x0)[1, :2], rel=1e-4)
    assert intAA.y(prob.x0)[2:]+int1A.y(prob.x0)[2:]+intA1.y(prob.x0)[2:] == pytest.approx(mix.y(prob.x0)[1, 2:], rel=1e-4)
    
    assert intAA.dydx(prob.x0)[:2]+int0A.dydx(prob.x0)[:2]+intA0.dydx(prob.x0)[0, :2] == pytest.approx(mix.dydx(prob.x0)[0, :2], rel=1e-4)
    assert intAA.dydx(prob.x0)[2:]+int0A.dydx(prob.x0)[2:]+intA1.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[0, 2:], rel=1e-4)
    assert intAA.dydx(prob.x0)[:2]+int1A.dydx(prob.x0)[:2]+intA0.dydx(prob.x0)[1, :2] == pytest.approx(mix.dydx(prob.x0)[1, :2], rel=1e-4)
    assert intAA.dydx(prob.x0)[2:]+int1A.dydx(prob.x0)[2:]+intA1.dydx(prob.x0)[2:] == pytest.approx(mix.dydx(prob.x0)[1, 2:], rel=1e-4)

    assert intAA.ddyddx(prob.x0)[:2]+int0A.ddyddx(prob.x0)[:2]+intA0.ddyddx(prob.x0)[0, :2] == pytest.approx(mix.ddyddx(prob.x0)[0, :2], rel=1e-4)
    assert intAA.ddyddx(prob.x0)[2:]+int0A.ddyddx(prob.x0)[2:]+intA1.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[0, 2:], rel=1e-4)
    assert intAA.ddyddx(prob.x0)[:2]+int1A.ddyddx(prob.x0)[:2]+intA0.ddyddx(prob.x0)[1, :2] == pytest.approx(mix.ddyddx(prob.x0)[1, :2], rel=1e-4)
    assert intAA.ddyddx(prob.x0)[2:]+int1A.ddyddx(prob.x0)[2:]+intA1.ddyddx(prob.x0)[2:] == pytest.approx(mix.ddyddx(prob.x0)[1, 2:], rel=1e-4)


if __name__ == "__main__":
    test_uniform(4)
    test_different_per_response(4)
    test_add_per_response(4)
    test_different_per_variable(4)
    test_add_per_variable(4)
    test_different_per_variable_and_response(4)
    test_add_per_variable_and_response(4)
    test_add_per_variable_and_response_multiple_overlap(4)


