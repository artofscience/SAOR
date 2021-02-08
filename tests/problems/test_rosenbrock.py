import pytest
import numpy as np

from sao.problems.rosenbrock import Rosenbrock


@pytest.mark.parametrize('n', [2, 5, 10])
def test_rosenbrock(n):
    problem = Rosenbrock(n)
    assert problem.objective.n == n

    x = np.ones(n)
    assert pytest.approx(problem.response(x), 0)
    assert problem.sensitivity(x).shape == (1, n)
