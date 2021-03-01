import pytest
import numpy as np

from Problems.square import Square


@pytest.mark.parametrize('n', [2, 5, 10])
def test_rosenbrock(n):
    problem = Square(n)
    assert problem.n == n

    x = np.ones(n)
    assert pytest.approx(problem.g(x), 0)
    assert problem.dg(x).shape == (2, n)
