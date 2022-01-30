from problems.n_dim.square import Square
from sao.mappings.change_of_variable import Exponential as Exp
from sao.mappings.change_of_variable import ConLin
import numpy as np
import pytest


def test_lin(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(1)

    assert mapping.g(x) == pytest.approx(x, tol)
    assert mapping.dg(x) == pytest.approx(1, tol)
    assert mapping.ddg(x) == pytest.approx(0, tol)


def test_rec(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(-1)

    assert mapping.g(x) == pytest.approx(1 / x, tol)
    assert mapping.dg(x) == pytest.approx(-1 / x ** 2, tol)
    assert mapping.ddg(x) == pytest.approx(2 / x ** 3, tol)


def test_lin_rec(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(Exp(-1), 1)

    assert mapping.g(x) == pytest.approx(1 / x, tol)
    assert mapping.dg(x) == pytest.approx(-1 / x ** 2, tol)
    assert mapping.ddg(x) == pytest.approx(2 / x ** 3, tol)


def test_exp2(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(2)
    assert mapping.g(x) == pytest.approx(x ** 2, tol)
    assert mapping.dg(x) == pytest.approx(2 * x, tol)
    assert mapping.ddg(x) == pytest.approx(2, tol)


def test_rec_lin(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(Exp(1), -1)

    assert mapping.g(x) == pytest.approx(1 / x, tol)
    assert mapping.dg(x) == pytest.approx(-1 / x ** 2, tol)
    assert mapping.ddg(x) == pytest.approx(2 / x ** 3, tol)


def test_rec_rec(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(Exp(-1), -1)
    assert mapping.g(x) == pytest.approx(x, tol)
    assert mapping.dg(x) == pytest.approx(1, tol)
    assert mapping.ddg(x) == pytest.approx(0, tol)


def test_rec_exp2_rec(tol=1e-4):
    x = np.array([1.0, 2.0])
    mapping = Exp(Exp(Exp(-1), 2), -1)
    assert mapping.g(x) == pytest.approx(Exp(2).g(x), tol)
    assert mapping.dg(x) == pytest.approx(Exp(2).dg(x), tol)
    assert mapping.ddg(x) == pytest.approx(Exp(2).ddg(x), tol)


def test_conlin(dx=1, tol=1e-4):
    prob = Square(10)
    df = prob.dg(prob.x0)
    conlin = ConLin()
    conlin.update(prob.x0, df)

    y = prob.x0 + dx

    assert conlin.g(y)[0, :] == pytest.approx(y, tol)
    assert conlin.g(y)[1, :] == pytest.approx(1 / y, tol)

    assert conlin.dg(y)[0, :] == pytest.approx(1, tol)
    assert conlin.dg(y)[1, :] == pytest.approx(-1 / y ** 2, tol)

    assert conlin.ddg(y)[0, :] == pytest.approx(0, tol)
    assert conlin.ddg(y)[1, :] == pytest.approx(2 / y ** 3, tol)


if __name__ == "__main__":
    test_conlin()
