from problems.n_dim.square import Square
from sao.mappings.approximations import LinearApproximation as LA
from sao.mappings.approximations import DiagonalQuadraticApproximation as DQA
from sao.mappings.change_of_variable import Exponential as Exp
from sao import intervening_variables, approximations
import numpy as np
import pytest


def test_ta(dx=1, tol=1e-4):
    prob = Square(4)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    old = approximations.Taylor1()
    old.update(x, f, df)

    new = LA()
    new.update(x, df)

    assert f + np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    assert new.ddg(x) == pytest.approx(old.ddg(x), tol)

    y = x + dx
    assert f + np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
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
    new = LA(Exp(p=1))
    new.update(x, df)

    assert f + np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    assert new.ddg(x) == pytest.approx(old.ddg(x), tol)

    y = x + dx
    assert f + np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


def test_ta_rec(dx=1, tol=1e-4):
    prob = Square(10)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    # Oldskool aka old
    old = approximations.Taylor1(intervening_variables.Exponential(p=-1))
    old.update(x, f, df)

    # Newskool aka new
    new = LA(Exp(p=-1))
    new.update(x, df)

    assert f + np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    assert new.ddg(x) == pytest.approx(old.ddg(x), tol)

    y = x + dx
    assert f + np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


def test_ta_ta_rec(dx=1, tol=1e-4):
    prob = Square(10)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    # Oldskool aka old
    old = approximations.Taylor1(intervening_variables.Exponential(p=2))
    old.update(x, f, df)

    # Newskool aka new
    new = LA(LA(Exp(p=2)))
    new.update(x, df)

    assert f + np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    assert new.ddg(x) == pytest.approx(old.ddg(x), tol)

    y = x + dx
    assert f + np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


def test_ta2(dx=1, tol=1e-4):
    prob = Square(4)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)
    ddf = prob.ddg(x)

    old = approximations.Taylor2()
    old.update(x, f, df, ddf)

    new = DQA()
    new.update(x, df, ddf)

    assert f + np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    assert new.ddg(x) == pytest.approx(old.ddg(x), tol)

    y = x + dx
    assert f + np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


def test_ta2_rec(dx=1, tol=1e-4):
    prob = Square(4)
    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)
    ddf = prob.ddg(x)

    old = approximations.Taylor2(intervening_variables.Exponential(p=-1))
    old.update(x, f, df, ddf)

    new = DQA(Exp(p=-1))
    new.update(x, df, ddf)

    assert f + np.sum(new.g(x), 1) == pytest.approx(old.g(x), tol)
    assert new.dg(x) == pytest.approx(old.dg(x), tol)
    assert new.ddg(x) == pytest.approx(old.ddg(x), tol)

    y = x + dx
    assert f + np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)
