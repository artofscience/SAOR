from problems.n_dim.square import Square
from sao.mappings.mapping import Taylor1 as Ta
from sao.mappings.mapping import Linear
from sao.mappings.mapping import Exponential as Exp
from sao import intervening_variables, approximations
import numpy as np
import pytest


def test_lin(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Linear()

    assert mapping.g(x) == pytest.approx(x, tol)
    assert mapping.dg(x) == pytest.approx(1, tol)
    assert mapping.ddg(x) == pytest.approx(0, tol)


def test_rec(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(p=-1)

    assert mapping.g(x) == pytest.approx(1 / x, tol)
    assert mapping.dg(x) == pytest.approx(-1 / x ** 2, tol)
    assert mapping.ddg(x) == pytest.approx(2 / x ** 3, tol)


def test_lin_rec(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(Exp(p=-1), p=1)

    assert mapping.g(x) == pytest.approx(1 / x, tol)
    assert mapping.dg(x) == pytest.approx(-1 / x ** 2, tol)
    assert mapping.ddg(x) == pytest.approx(2 / x ** 3, tol)


def test_exp2(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(p=2)
    assert mapping.g(x) == pytest.approx(x ** 2, tol)
    assert mapping.dg(x) == pytest.approx(2 * x, tol)
    assert mapping.ddg(x) == pytest.approx(2, tol)


def test_rec_lin(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(Exp(p=1), p=-1)

    assert mapping.g(x) == pytest.approx(1 / x, tol)
    assert mapping.dg(x) == pytest.approx(-1 / x ** 2, tol)
    assert mapping.ddg(x) == pytest.approx(2 / x ** 3, tol)


def test_rec_rec(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(Exp(p=-1), p=-1)
    assert mapping.g(x) == pytest.approx(x, tol)
    assert mapping.dg(x) == pytest.approx(1, tol)
    assert mapping.ddg(x) == pytest.approx(0, tol)


def test_rec_exp2_rec(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(Exp(Exp(p=-1), p=2), p=-1)
    assert mapping.g(x) == pytest.approx(Exp(p=2).g(x), tol)
    assert mapping.dg(x) == pytest.approx(Exp(p=2).dg(x), tol)
    assert mapping.ddg(x) == pytest.approx(Exp(p=2).ddg(x), tol)


def test_ta(dx=1, tol=1e-4):
    prob = Square(4)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    old = approximations.Taylor1()
    old.update(x, f, df)

    new = Ta()
    new.update(x, f, df)

    assert np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    assert new.ddg(x) == pytest.approx(old.ddg(x), tol)

    y = x + dx
    assert np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


def test_ta_lin(dx=1, tol=1e-4):
    prob = Square(4)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    # Oldskool aka old
    old = approximations.Taylor1(intervening_variables.Exponential(p=1))
    old.update(x, f, df)

    # Newskool aka new
    new = Ta(Exp(p=1))
    new.update(x, f, df)

    assert np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    assert new.ddg(x) == pytest.approx(old.ddg(x), tol)

    y = x + dx
    assert np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


def test_ta_rec(dx=1, tol=1e-4):
    prob = Square(10)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    # Oldskool aka old
    old = approximations.Taylor1(intervening_variables.Exponential(p=2))
    old.update(x, f, df)

    # Newskool aka new
    new = Ta(Exp(p=2))
    new.update(x, f, df)

    # NOT WORKING
    assert np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    assert new.ddg(x) == pytest.approx(old.ddg(x), tol)


if __name__ == "__main__":
    test_lin()
    test_rec()
    test_exp2()
    test_lin_rec()
    test_rec_lin()
    test_rec_rec()
    test_rec_exp2_rec()
    test_ta()
    test_ta_lin()
    test_ta_rec()
