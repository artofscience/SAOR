import numpy as np
import logging
from Problems.topology_optimization_benchmark.Mbb_beam import MBBBeam
from sao.approximations.taylor import Taylor1
from sao.approximations.intervening import MMA
from sao.move_limits.ml_intervening import MoveLimitIntervening
from sao.problems.subproblem import Subproblem
from sao.solvers.interior_point import InteriorPointXYZ as ipopt
# from line_profiler import LineProfiler

np.set_printoptions(precision=4)

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# # If you want to write to a .log file (stored in the same directory as the script you run)
# file_handler = logging.FileHandler('test_mixed_square.log')
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


def test_mbb(nelx=40, nely=20, volfrac=0.4, penal=3, rmin=2):

    # Instantiate problem
    prob = MBBBeam(nelx, nely, volfrac, penal, rmin)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None
    solves = 0

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

        solver = ipopt(subprob, x0=x_k)
        x_k = solver.update()
        solves += solver.itera

        itte += 1

    logger.info('Optimization loop converged!')
    print(solves)


if __name__ == "__main__":
    test_mbb()
