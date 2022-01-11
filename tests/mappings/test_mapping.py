from problems.n_dim.square import Square
from sao.mappings.mapping import Taylor1 as Ta
from sao.mappings.mapping import Exponential as Exp
from sao import intervening_variables, approximations
import numpy as np
import pytest


def test_exp_exp(dx=1, tol=1e-4):
    prob = Square(4)
    x = prob.x0
    mapping = Exp(Exp(p=-1), p=-1)
    assert mapping.g(x) == pytest.approx(x, tol)
    assert mapping.dg(x) == pytest.approx(1, tol)
    assert mapping.ddg(x) == pytest.approx(0, tol)

    y = x + dx
    assert mapping.g(y) == pytest.approx(y, tol)


def test_ta(dx=1, tol=1e-4):
    prob = Square(4)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    old = approximations.Taylor1()
    old.update(x, f, df)

    new = Ta()
    new.update(x, f, df)

    y = x + dx
    assert np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


def test_ta_exp(dx=1, tol=1e-4):
    prob = Square(4)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    # Oldskool aka old
    old = approximations.Taylor1(intervening_variables.Exponential(p=-1))
    old.update(x, f, df)

    # Newskool aka new
    new = Ta(Exp(p=-1))
    new.update(x, f, df)

    y = x + dx
    assert np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


def test_ta_exp_exp(dx=1, tol=1e-4):
    prob = Square(4)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    old = approximations.Taylor1()
    old.update(x, f, df)

    new = Ta(Exp(Exp(p=-1), p=-1))
    new.update(x, f, df)

    assert np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    # assert new.ddg(x) == pytest.approx(old.ddg(x), tol)

    y = x + dx
    assert np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    # assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


if __name__ == "__main__":
    test_exp_exp()
    test_ta()
    test_ta_exp()
    test_ta_exp_exp()
