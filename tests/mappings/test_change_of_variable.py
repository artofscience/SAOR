from problems.n_dim.square import Square
from sao.mappings.change_of_variable import Exponential as Exp
from sao.mappings.change_of_variable import ConLin
import numpy as np
import pytest


def mapping_test(mapping, x, g, dg, ddg, tol=1e-4):
    assert mapping.g(x) == pytest.approx(g, tol)
    assert mapping.dg(x) == pytest.approx(dg, tol)
    assert mapping.ddg(x) == pytest.approx(ddg, tol)


class TestExp:
    """
    Test simple exponential function give correct function values and derivatives.
    """

    x = np.array([1.0, 2.0, 3.0])  # Note this only tests for simple x; try some more

    # Tests fails for a variable that takes zero value!
    # Exp(0) should raise error.

    def test_lin(self): mapping_test(Exp(1), self.x, self.x, 1, 0)

    def test_exp2(self): mapping_test(Exp(2), self.x, self.x ** 2, 2 * self.x, 2)

    def test_rec(self): mapping_test(Exp(-1), self.x, 1 / self.x, -1 / self.x ** 2, 2 / self.x ** 3)

    @pytest.mark.parametrize('i', [-3, -2, -1, 1, 2, 3])
    def test_exp(self, i): mapping_test(Exp(i), self.x, self.x ** i, i * self.x ** (i - 1),
                                        i * (i - 1) * self.x ** (i - 2))


class TestStackedExp:
    """
    Test stacked exponential functions give correct function values and derivatives.
    """

    x = np.array([1.0, 2.0, 3.0])  # Note this only tests for simple x; try some more

    def test_lin_rec(self): mapping_test(Exp(1, Exp(-1)), self.x, 1 / self.x, -1 / self.x ** 2, 2 / self.x ** 3)

    def test_rec_lin(self): mapping_test(Exp(-1, Exp(1)), self.x, 1 / self.x, -1 / self.x ** 2, 2 / self.x ** 3)

    def test_rec_rec(self): mapping_test(Exp(-1, Exp(-1)), self.x, self.x, 1, 0)

    def test_rec_exp2_rec(self):
        map = Exp(-1, Exp(2, Exp(-1)))
        map2 = Exp(2)
        mapping_test(map, self.x, map2.g(self.x), map2.dg(self.x), map2.ddg(self.x))  # Requires some comments


class TestTwoMap:
    def test_conlin(self, tol=1e-4):
        problem = Square(10)  # Also test different initial values
        df = problem.dg(problem.x0)
        conlin = ConLin()
        conlin.update(problem.x0, df)

        dx = 1
        y = problem.x0 + dx

        assert conlin.g(y)[0, :] == pytest.approx(y, tol)
        assert conlin.g(y)[1, :] == pytest.approx(1 / y, tol)

        assert conlin.dg(y)[0, :] == pytest.approx(1, tol)
        assert conlin.dg(y)[1, :] == pytest.approx(-1 / y ** 2, tol)

        assert conlin.ddg(y)[0, :] == pytest.approx(0, tol)
        assert conlin.ddg(y)[1, :] == pytest.approx(2 / y ** 3, tol)
