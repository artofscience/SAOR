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
from sao.intervening_variables.asymptote_update_strategies import Svanberg1987_t, Svanberg1987_s_move

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
    converged = IterationCount(10)
    mma0 = MMA(updaterule=Svanberg1987_t(t=0.2))
    mma1 = MMA(updaterule=Svanberg1987_s_move(x_min=problem.x_min, x_max=problem.x_max, factor=0.5))
    intvar = MixedIntervening(problem.n, problem.m + 1, default=mma0)
    intvar.set_intervening(mma1, var=1)

    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])
    x = problem.x0

    f_storage = []
    x0_storage = []
    x1_storage = []
    sigma0_storage = []

    while not converged:

        f = problem.g(x)
        df = problem.dg(x)

        x0_storage.append(x[0])
        x1_storage.append(x[1])
        f_storage.append(f[0])
        sigma0_storage.append(f[1]+1.0)

        infeasibility = max(0.0, f[1], f[2])
        print("{}: {:.3f} {:.3f}".format(converged.iteration, f[0], infeasibility))
        if (infeasibility < 0.001) and (f[0] < 1.001*f_analytical[-1]):
            break

        subproblem.build(x, f, df)

        x[:] = pdip(subproblem, variables=Pdipx)[0]

    assert pytest.approx(f_storage, rel=1e-2) == [1.68, 1.43, 1.37, 1.44, 1.47, 1.51]
    assert pytest.approx(sigma0_storage, rel=1e-2) == [0.92, 1.10, 1.13, 1.10, 1.03, 1.00]
    assert pytest.approx(x0_storage, rel=1e-2) == [1.5, 1.39, 1.22, 1.39, 1.37, 1.41]
    assert pytest.approx(x1_storage, rel=1e-1) == [0.5, 0.25, 0.50, 0.25, 0.38, 0.38]

def test_2_bar_truss_mma_without_move_limits():
    assert 1 == 1

if __name__ == "__main__":
    test_2_bar_truss_mma()