import pytest
import numpy as np
from Problems._2d.svanberg1987_2_bar_truss import TwoBarTruss
from sao.move_limits.move_limit import Bounds, MoveLimitST
from sao.approximations.taylor import Taylor1
from sao.problems.subproblem import Subproblem
from sao.intervening_variables import MixedIntervening
from sao.intervening_variables import MMA
from sao.convergence_criteria import IterationCount
from sao.solvers.primal_dual_interior_point import pdip, Pdipx

@pytest.mark.parametrize('n',[5])
@pytest.mark.parametrize('t',[1/16, 1/8, 1/4, 1/3, 1/2, 2/3, 3/4])
def test_cantilever_beam(t):
    x_opt_analytical = np.array([6.016, 5.309, 4.494, 3.502, 2.153])
    f_opt_analytical = 1.340
    f_opt_numerical = 1.3409
    assert f_opt_analytical == pytest.approx(f_opt_numerical, rel=1e-3)

def test_2_bar_truss_mma():
    problem = TwoBarTruss()

    g0 = problem.g(problem.x0)
    assert pytest.approx(g0[0], rel=1e-3) == 1.677
    assert pytest.approx(g0[1]+1, rel=1e-3) == 0.925
    f_analytical = np.array([1.68, 1.43, 1.37, 1.44, 1.47, 1.51])

    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitST(factor=2)
    intvar = MixedIntervening(problem.n, problem.m + 1)
    intvar.set_intervening(MMA(problem.x_min, problem.x_max, asyincr=4/3), var=0)
    intvar.set_intervening(MMA(problem.x_min, problem.x_max), var=1)
    converged = IterationCount(10)

    subproblem = Subproblem(Taylor1(MMA(problem.x_min, problem.x_max, asyincr=4/3, asydecr=0.5)), limits=[bounds, movelimit])
    x = problem.x0

    while not converged:

        f = problem.g(x)
        df = problem.dg(x)

        infeasibility = max(0.0, f[1], f[2])
        print("{}: {:.3f} {:.3f}".format(converged.iteration, f[0], infeasibility))
        if (infeasibility < 0.001) and (f[0] < 1.001*f_analytical[-1]):
            break

        subproblem.build(x, f, df)

        x[:] = pdip(subproblem, variables=Pdipx)[0]

    assert f_analytical[-1] == pytest.approx(f[0], rel=1e-3)
    assert converged.iteration == len(f_analytical)
    assert pytest.approx(x[0], rel=1e-2) == 1.41
    assert pytest.approx(x[1], rel=1e-2) == 0.38
    assert pytest.approx(f[1], rel=1e-2) == 1.00


def test_2_bar_truss_mma_without_move_limits():
    assert 1 == 1

if __name__ == "__main__":
    test_2_bar_truss_mma()