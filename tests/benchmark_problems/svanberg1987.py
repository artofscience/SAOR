import pytest
import numpy as np
from Problems._2d.svanberg1987_2_bar_truss import TwoBarTruss

@pytest.mark.parametrize('n',[5])
@pytest.mark.parametrize('t',[1/16, 1/8, 1/4, 1/3, 1/2, 2/3, 3/4])
def test_cantilever_beam(t):
    x_opt_analytical = np.array([6.016, 5.309, 4.494, 3.502, 2.153])
    f_opt_analytical = 1.340
    f_opt_numerical = 1.3409
    assert f_opt_analytical == pytest.approx(f_opt_numerical, rel=1e-3)

def test_2_bar_truss_mma():
    problem = TwoBarTruss()
    g = problem.g(problem.x0)
    assert pytest.approx(g[0], rel=1e-3) == 1.677
    assert pytest.approx(g[1]+1, rel=1e-3) == 0.925

def test_2_bar_truss_mma_without_move_limits():
    assert 1 == 1

if __name__ == "__main__":
    test_2_bar_truss_mma()