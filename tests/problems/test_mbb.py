import numpy as np
import logging
from Problems.topology_optimization_benchmark.compliance import Compliance
from Problems.topology_optimization_benchmark.stress import Stress
from Problems.topology_optimization_benchmark.mechanism import Mechanism
from Problems.topology_optimization_benchmark.eigenvalue import Eigenvalue
from sao.approximations.taylor import Taylor1
from sao.intervening_vars.intervening import MMA
from sao.move_limits.ml_intervening import MoveLimitIntervening
from sao.problems.subproblem import Subproblem
from sao.solvers.interior_point import InteriorPointXYZ as ipopt
from sao.util.plotter import Plot, Plot2
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


def test_compliance(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3):

    # Instantiate problem
    prob = Compliance(nelx, nely, volfrac, penal, rmin)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None
    solves = 0

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2 = Plot2(prob, responses=np.array([0]), variables=np.arange(0, prob.n, 50))

    # Optimization loop
    while itte < 100:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Print current iteration and x_k
        vis = prob.visualize(x_k, itte, vis)
        logger.info('iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  vol: {:>6.3f}'.format(
            itte, f[0], f[1], np.mean(np.asarray(prob.H * x_k[np.newaxis].T / prob.Hs)[:, 0])))
        plotter.plot([f[0], f[1]])

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Plot current approximation
        plotter2.plot_approx(x_k, f, prob, subprob)

        solver = ipopt(subprob, x0=x_k)
        x_k = solver.update()
        solves += solver.itera

        itte += 1

    logger.info('Optimization loop converged!')
    print(solves)


def test_stress(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=2, max_stress=1):
    # Instantiate problem
    prob = Stress(nelx, nely, volfrac, penal, rmin, max_stress=max_stress)
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
    while itte < 1000:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Print current iteration and x_k
        vis = prob.visualize_field(prob.stress, max_stress, itte, vis)
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


def test_mechanism(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3,kin=0.01, kout=0.01):

    # Instantiate problem
    prob = Mechanism(nelx, nely, volfrac, penal, rmin, kin=kin, kout=kout)
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


def test_eigenvalue(nelx=200, nely=50, volfrac=0.6, penal=3, rmin=3):

    # Instantiate problem
    prob = Eigenvalue(nelx, nely, volfrac, penal, rmin)
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
    test_compliance(nelx=20, nely=10, volfrac=0.3)
    test_stress(nelx=200, nely=50, max_stress=1)
    test_mechanism(nelx=200, nely=100, kin=0.0005, kout=0.0005, volfrac=0.3)
    test_eigenvalue()
