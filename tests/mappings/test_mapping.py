from problems.n_dim.square import Square
from sao.mappings.mapping import LinearApproximation as LA
from sao.mappings.mapping import DiagonalQuadraticApproximation as DQA
from sao.mappings.mapping import Exponential as Exp
from sao.mappings.mapping import ConLin
from sao import intervening_variables, approximations
import numpy as np
import pytest


def test_lin(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(p=1)

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


def test_conlin(dx=1, tol=1e-4):
    prob = Square(10)
    f = prob.g(prob.x0)
    df = prob.dg(prob.x0)
    inter = ConLin()
    inter.update(prob.x0, df)
    lin = Exp(p=1)
    rec = Exp(p=-1)

    # Check g(x) for ConLin, Linear(), and Reciprocal()
    assert inter.g(prob.x0)[0, :] == pytest.approx(lin.g(prob.x0), rel=1e-4)
    assert inter.g(prob.x0)[1, :] == pytest.approx(rec.g(prob.x0), rel=1e-4)
    assert lin.g(prob.x0) == pytest.approx(prob.x0, rel=1e-4)
    assert rec.g(prob.x0) == pytest.approx(1 / prob.x0, rel=1e-4)

    assert inter.dg(prob.x0)[0, :] == pytest.approx(lin.dg(prob.x0), rel=1e-4)
    assert inter.dg(prob.x0)[1, :] == pytest.approx(rec.dg(prob.x0), rel=1e-4)
    assert lin.dg(prob.x0) == pytest.approx(np.ones_like(prob.x0), rel=1e-4)
    assert rec.dg(prob.x0) == pytest.approx(-1 / prob.x0 ** 2, rel=1e-4)

    assert inter.ddg(prob.x0)[0, :] == pytest.approx(lin.ddg(prob.x0), rel=1e-4)
    assert inter.ddg(prob.x0)[1, :] == pytest.approx(rec.ddg(prob.x0), rel=1e-4)
    assert lin.ddg(prob.x0) == pytest.approx(np.zeros_like(prob.x0), abs=1e-4)
    assert rec.ddg(prob.x0) == pytest.approx(2 / prob.x0 ** 3, rel=1e-4)


if __name__ == "__main__":
    test_conlin()
