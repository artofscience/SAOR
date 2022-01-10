from problems.n_dim.square import Square
from sao.mappings.mapping import Taylor1 as Ta
from sao.mappings.mapping import Exponential as Exp
from sao import intervening_variables, approximations
import numpy as np
import pytest


def test_mapping(dx=1, tol=1e-4):
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

    old2 = approximations.Taylor1()
    old2.update(x, f, df)

    new2 = Ta()
    new2.update(x, f, df)

    assert np.sum(new2.g(y), 1) == pytest.approx(old2.g(y), tol)
    assert new2.dg(y) == pytest.approx(old2.dg(y), tol)
    assert new2.ddg(y) == pytest.approx(old2.ddg(y), tol)

    mapping = Exp(Exp(p=-1), p=-1)
    assert mapping.g(x) == pytest.approx(x, tol)
    assert mapping.g(y) == pytest.approx(y, tol)

    map1 = Exp(p=-1)
    map2 = Exp(map1, p=-1)
    new3 = Ta(map2)
    new3.update(x, f, df)

    assert np.sum(new3.g(x), 1) == pytest.approx(old2.g(x), tol)
    assert new3.dg(x) == pytest.approx(old2.dg(x), tol)

    # assert np.sum(new3.g(y), 1) == pytest.approx(old2.g(y), tol)
    # assert new3.ddg(x) == pytest.approx(old2.ddg(x), tol)


if __name__ == "__main__":
    test_mapping(1)
