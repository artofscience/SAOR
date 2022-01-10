from problems.n_dim.square import Square
from sao.mappings import mapping
from sao import intervening_variables, approximations
import numpy as np
import pytest


def test_mapping(dx=1, tol=1e-6):
    prob = Square(4)

    x = prob.x0
    f = prob.g(x)
    df = prob.dg(x)

    # Oldskool aka old
    old = approximations.Taylor1(intervening_variables.Exponential(-1))
    old.update(x, f, df)

    # Newskool aka new
    new = mapping.Taylor1(mapping.Exponential(-1))
    new.update(x, f, df)

    y = x + dx
    assert np.sum(new.g(y), 1) == pytest.approx(old.g(y), tol)
    assert new.dg(y) == pytest.approx(old.dg(y), tol)
    assert new.ddg(y) == pytest.approx(old.ddg(y), tol)


if __name__ == "__main__":
    test_mapping(1)
