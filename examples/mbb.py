import numpy as np
import logging
from Problems.topology_optimization_benchmark.compliance import Compliance
from Problems.topology_optimization_benchmark.stress import Stress
from Problems.topology_optimization_benchmark.mechanism import Mechanism
from Problems.topology_optimization_benchmark.eigenvalue import Eigenvalue
from sao.approximations.taylor import Taylor1, SphericalTaylor2, NonSphericalTaylor2
from sao.intervening_vars.intervening import Linear, ConLin, MMA
from sao.move_limits.move_limit import MoveLimitIntervening, MoveLimit1
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


def example_compliance(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3):
    logger.info("Solving compliance minimization subject to volume constraint")

    # Instantiate problem
    prob = Compliance(nelx, nely, volfrac, penal, rmin)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=NonSphericalTaylor2(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    # subprob = Subproblem(intervening=Linear(), approximation=NonSphericalTaylor2(),
    #                      ml=MoveLimit1(xmin=prob.xmin, xmax=prob.xmax))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None
    solves = 0

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0]), variables=np.arange(3, prob.n, 100))


    # Optimization loop
    while itte < 250:

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
        if plotter2_flag:
            plotter2.plot_approx(x_k, f, prob, subprob, itte)

        solver = ipopt(subprob, x0=x_k)
        x_k = solver.update()
        solves += solver.itera

        itte += 1

    logger.info('Optimization loop converged!')
    print(solves)


def example_stress(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=2, max_stress=1):
    logger.info("Solving volume minimization subject to aggregated stress constraint")

    # Instantiate problem
    prob = Stress(nelx, nely, volfrac, penal, rmin, max_stress=max_stress)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    # subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=NonSphericalTaylor2(),
    #                      ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob = Subproblem(intervening=Linear(), approximation=NonSphericalTaylor2(),
                         ml=MoveLimit1(xmin=prob.xmin, xmax=prob.xmax))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None
    solves = 0

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([1]), variables=np.arange(3, prob.n, 100))

    # Optimization loop
    while itte < 500:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Print current iteration and x_k
        vis = prob.visualize_field(prob.stress, max_stress, itte, vis)
        logger.info('iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  vol: {:>6.3f}'.format(
            itte, f[0], f[1], np.mean(np.asarray(prob.H * x_k[np.newaxis].T / prob.Hs)[:, 0])))
        plotter.plot([f[0], f[1]])

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_approx(x_k, f, prob, subprob, itte)

        solver = ipopt(subprob, x0=x_k)
        x_k = solver.update()
        solves += solver.itera

        itte += 1

    logger.info('Optimization loop converged!')
    print(solves)


def example_mechanism(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3, kin=0.01, kout=0.01):
    logger.info("Solving displacement minimization subject to volume constraint")

    # Instantiate problem
    prob = Mechanism(nelx, nely, volfrac, penal, rmin, kin=kin, kout=kout)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    # subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=NonSphericalTaylor2(),
    #                      ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob = Subproblem(intervening=Linear(), approximation=NonSphericalTaylor2(),
                         ml=MoveLimit1(xmin=prob.xmin, xmax=prob.xmax))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None
    solves = 0

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0]), variables=np.arange(3, prob.n, 100))

    # Optimization loop
    while itte < 250:

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
        if plotter2_flag:
            plotter2.plot_approx(x_k, f, prob, subprob, itte)

        solver = ipopt(subprob, x0=x_k)
        x_k = solver.update()
        solves += solver.itera

        itte += 1

    logger.info('Optimization loop converged!')
    print(solves)


def example_eigenvalue(nelx=100, nely=50, volfrac=0.6, penal=3, rmin=3):
    logger.info("Solving eigenfrequency maximization subject to volume constraint")

    # Instantiate problem
    prob = Eigenvalue(nelx, nely, volfrac, penal, rmin)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    # subprob = Subproblem(intervening=Linear(), approximation=Taylor1(),
    #                      ml=MoveLimit1(xmin=prob.xmin, xmax=prob.xmax))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None
    solves = 0

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0]), variables=np.arange(9, prob.n, 60))

    # Optimization loop
    while itte < 250:

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
        if plotter2_flag:
            plotter2.plot_approx(x_k, f, prob, subprob, itte)

        solver = ipopt(subprob, x0=x_k)
        x_k = solver.update()
        solves += solver.itera

        itte += 1

    logger.info('Optimization loop converged!')
    print(solves)


if __name__ == "__main__":
    # example_compliance(nelx=100, nely=50, volfrac=0.3)                            # use nelx=50, nely=20 for plotter2
    # example_stress(nelx=100, nely=50, max_stress=1)                               # use nelx=50, nely=20 for plotter2
    # example_mechanism(nelx=100, nely=50, kin=0.0005, kout=0.0005, volfrac=0.3)    # use nelx=50, nely=20 for plotter2
    example_eigenvalue(nelx=50, nely=20, volfrac=0.6, penal=3, rmin=3)           # use nelx=50, nely=20 for plotter2
