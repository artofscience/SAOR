import pytest
import numpy as np
from Problems.svanberg1987 import CantileverBeam, TwoBarTruss
from sao.move_limits.move_limit import Bounds, MoveLimitST
from sao.approximations.taylor import Taylor1
from sao.problems.subproblem import Subproblem
from sao.intervening_variables import MixedIntervening
from sao.intervening_variables import MMA
from sao.convergence_criteria import IterationCount
from sao.solvers.primal_dual_interior_point import pdip, Pdipx, Pdipxy, Pdipxyz
from sao.intervening_variables.asymptote_update_strategies import Svanberg1987_t, Svanberg1987_s_move

@pytest.mark.parametrize('pdiptype', [Pdipx, Pdipxy, Pdipxyz])
def test_cantilever_beam_mma(pdiptype):
    f_analytical = [1.560, 1.285, 1.307, 1.331, 1.337, 1.339, 1.340]
    problem = CantileverBeam()
    movelimit = MoveLimitST(factor=2)
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)

    g0 = problem.g(problem.x0)
    assert pytest.approx(g0[0], rel=1e-3) == 1.560
    assert pytest.approx(g0[1]+1, rel=1e-3) == 1.000

    converged = IterationCount(20)
    subproblem = Subproblem(Taylor1(MMA(updaterule=Svanberg1987_t(t=1/8))),
                            limits=[bounds, movelimit])
    x = problem.x0

    f_storage = []
    sigma0_storage = []

    while not converged:

        f = problem.g(x)
        df = problem.dg(x)

        f_storage.append(f[0])
        sigma0_storage.append(f[1] + 1.0)

        infeasibility = max(0.0, f[1])
        print("{}: {:.4f} {:.4f}".format(converged.iteration, f[0], infeasibility))
        if (infeasibility < 0.001) and (f[0] < 1.001*problem.f_opt):
            break

        subproblem.build(x, f, df)

        x[:] = pdip(subproblem, variables=pdiptype)[0]

    assert pytest.approx(f_storage, rel=1e-3) == f_analytical
    assert pytest.approx(x, rel=1e-1) == problem.x_opt
    assert pytest.approx(sigma0_storage, rel=1e-2) == [1.000, 1.23, 1.11, 1.03, 1.008, 1.002, 1.001]

@pytest.mark.parametrize('type', [Pdipx, Pdipxy, Pdipxyz])
def test_2_bar_truss_mma(pdiptype):
    problem = TwoBarTruss()

    g0 = problem.g(problem.x0)
    assert pytest.approx(g0[0], rel=1e-3) == 1.677
    assert pytest.approx(g0[1]+1, rel=1e-3) == 0.925
    f_analytical = [1.68, 1.43, 1.37, 1.44, 1.47, 1.51]

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
        if (infeasibility < 0.001) and (f[0] < 1.001*problem.f_opt):
            break

        subproblem.build(x, f, df)

        x[:] = pdip(subproblem, variables=pdiptype)[0]

    assert pytest.approx(f_storage, rel=1e-2) == f_analytical
    assert pytest.approx(sigma0_storage, rel=1e-2) == [0.92, 1.10, 1.13, 1.10, 1.03, 1.00]
    assert pytest.approx(x0_storage, rel=1e-2) == [1.5, 1.39, 1.22, 1.39, 1.37, 1.41]
    assert pytest.approx(x1_storage, rel=1e-1) == [0.5, 0.25, 0.50, 0.25, 0.38, 0.38]

if __name__ == "__main__":
    test_2_bar_truss_mma(Pdipx)
    test_2_bar_truss_mma(Pdipxy)
    test_2_bar_truss_mma(Pdipxyz)

    test_cantilever_beam_mma(Pdipx)
    test_cantilever_beam_mma(Pdipxy)
    test_cantilever_beam_mma(Pdipxyz)
