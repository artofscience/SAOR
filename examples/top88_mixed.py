import numpy as np
import logging
from Problems.topology_optimization_benchmark.mbbbeam import Top88
from sao.approximations.taylor import Taylor1
from sao.intervening_vars.intervening import Linear
from sao.move_limits.move_limit import MoveLimitIntervening
from sao.problems.subproblem import Subproblem
from sao.problems.mixed import Mixed
from sao.solvers.SolverIP_Svanberg import SvanbergIP

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

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
    subprob_map = {(0, 0): Subproblem(intervening=MMA(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]])),
                   (1, 0): Subproblem(intervening=Linear(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]]))}

    # Instantiate a mixed scheme
    subprob = Mixed(subprob_map, var_set, resp_set)

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Print current iteration and x_k
        vis = prob.visualize(x_k, itte, vis)
        logger.info('iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  vol: {:>6.3f}'.format(
            itte, f[0], f[1], np.mean(np.asarray(prob.H * x_k[np.newaxis].T / prob.Hs)[:, 0])))

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Call solver (x_k, g and dg are within approx instance)
        x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)
        x_k = x.copy()

        # solver = ipb(subprob, epsimin=1e-7)
        # solver.update()
        # x_k = solver.x.copy()

        itte += 1

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    test_mixed_top88()
