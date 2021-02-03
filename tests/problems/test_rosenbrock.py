import pytest

from sao.problems.rosenbrock import Rosenbrock


def test_rosenbrock():
    problem = Rosenbrock()
    assert pytest.approx(problem.response((1, 1)), 0)
