import pytest
import numpy as np
from Problems.Top88 import Top88
from sao.approximations.taylor import Taylor1, Taylor2
from sao.approximations.intervening import Linear, ConLin, MMA, ReciSquared
from sao.move_limits.ml_intervening import MoveLimitIntervening
from sao.problems.subproblem import Subproblem
from sao.problems.mixed import Mixed
from sao.solvers.interior_point_basis import InteriorPointBasis as ipb
from sao.solvers.interior_point_artificial import InteriorPointArtificial as ipa
from sao.solvers.SolverIP_Svanberg import SvanbergIP
from line_profiler import LineProfiler

np.set_printoptions(precision=4)


def test_mixed_top88(nelx=180, nely=60, volfrac=0.4, penal=3, rmin=5.4, ft=1):

    # Instantiate problem
    prob = Top88(nelx, nely, volfrac, penal, rmin, ft)
    assert prob.n == nelx * nely

    # Define variable and response sets as dictionaries
    var_set = {0: np.arange(0, prob.n)}
    resp_set = {0: np.array([0]),
                1: np.array([1])}

    # Instantiate a mixed approximation scheme
    # subprob_dict = {(0, 0): Subproblem(intervening=MMA(prob.xmin[var_set[0]], prob.xmax[var_set[0]]),
    #                                    approximation=Taylor1(),
    #                                    ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
    #                                                            xmax=prob.xmax[var_set[0]])),
    #                 (1, 0): Subproblem(intervening=Linear(),
    #                                    approximation=Taylor1(),
    #                                    ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
    #                                                            xmax=prob.xmax[var_set[0]]))}
    subprob_dict = {(0, 0): Subproblem(intervening=ReciSquared(),
                                       approximation=Taylor1(),
                                       ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                               xmax=prob.xmax[var_set[0]])),
                    (1, 0): Subproblem(intervening=Linear(),
                                       approximation=Taylor1(),
                                       ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                               xmax=prob.xmax[var_set[0]]))}

    # Instantiate a mixed scheme
    subprob = Mixed(prob.n, prob.m, subprob_dict, var_set, resp_set)

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    xold1 = np.zeros_like(x_k)
    vis = None

    # Optimization loop
    # while itte < 5:
    while np.any(x_k - xold1) > 1e-1:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Print current iteration and x_k
        vis = prob.visualize(x_k, itte, vis)
        print('iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  vol: {:>6.3f}'.format(
            itte, f[0], f[1], np.mean(np.asarray(prob.H * x_k[np.newaxis].T / prob.Hs)[:, 0])))

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Call solver (x_k, g and dg are within approx instance)
        x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)
        xold1 = x_k.copy()
        x_k = x.copy()

        # solver = ipb(subprob, epsimin=1e-7)
        # solver.update()
        # x_k = solver.x.copy()

        itte += 1

    print('Alles goed!')


if __name__ == "__main__":
    test_mixed_top88()