import numpy as np
import logging
from Problems.Square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.intervening_variables import Linear, ConLin, MMA
from sao.move_limits.move_limit import MoveLimitIntervening, MoveLimitMMA
from sao.problems.subproblem import Subproblem
from sao.problems.mixed import Mixed
from sao.solvers.interior_point import InteriorPointX as ipx
from sao.solvers.interior_point import InteriorPointXY as ipxy
from sao.solvers.interior_point import InteriorPointXYZ as ipxyz
from sao.solvers.SolverIP_Svanberg import SvanbergIP
from sao.util.plotter import Plot, Plot2
from sao.convergence_criteria.ObjChange import ObjectiveChange
from sao.convergence_criteria.VarChange import VariableChange
from sao.convergence_criteria.KKT import KKT
from sao.convergence_criteria.Feasibility import Feasibility
from sao.convergence_criteria.Alltogether import Alltogether

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

np.set_printoptions(precision=4)


def example_square_Svanberg(n):
    logger.info("Solving Square.py using Ipopt Svanberg")

    # Instantiate problem
    prob = Square(n)

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax, move_limit=1.0))

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0, 1]), variables=np.arange(0, prob.n))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Print current iteration and x_k
        logger.info('iter: {:^4d}  |  x: {:<20s}  |  obj: {:^9.3f}  |  constr: {:^6.3f}'.format(
            itte, np.array2string(x_k[0:2]), f[0], f[1]))
        plotter.plot([f[0], f[1]])

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        itte += 1

    logger.info('Alles goed!')


def example_square_ipx(n):
    logger.info("Solving test_square using Ipopt x")

    # Instantiate problem
    prob = Square(n)

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=Linear(), approximation=Taylor1(),
                         ml=MoveLimitMMA(xmin=prob.xmin, xmax=prob.xmax))

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0]), variables=np.arange(0, prob.n, 2))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Print current iteration and x_k
        logger.info('iter: {:^4d}  |  x: {:<20s}  |  obj: {:^9.3f}  |  constr: {:^6.3f}'.format(
            itte, np.array2string(x_k[0:2]), f[0], f[1]))
        plotter.plot([f[0], f[1]])

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        solver = ipx(subprob, epsimin=1e-6)
        x_k = solver.update()
        print(solver.iterin)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=solver.w.lam, df=df)

        itte += 1

    logger.info('Alles goed!')


def example_square_ipxy(n):
    logger.info("Solving test_square using Ipopt xy")

    # Instantiate problem
    prob = Square(n)
    assert prob.n == n

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=ConLin(), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0]), variables=np.arange(0, prob.n, 2))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Print current iteration and x_k
        logger.info('iter: {:^4d}  |  x: {:<20s}  |  obj: {:^9.3f}  |  constr: {:^6.3f}'.format(
            itte, np.array2string(x_k[0:2]), f[0], f[1]))
        plotter.plot([f[0], f[1]])

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        solver = ipxy(subprob, epsimin=1e-6)
        x_k = solver.update()
        print(solver.iterin)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=solver.w.lam, df=df)

        itte += 1

    logger.info('Alles goed!')


def example_square_ipxyz(n):
    logger.info("Solving test_square using Ipopt xyz")

    # Instantiate problem
    prob = Square(n)

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=Linear(), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0]), variables=np.arange(0, prob.n, 2))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Print current iteration and x_k
        logger.info('iter: {:^4d}  |  x: {:<20s}  |  obj: {:^9.3f}  |  constr: {:^6.3f}'.format(
            itte, np.array2string(x_k[0:2]), f[0], f[1]))
        plotter.plot([f[0], f[1]])

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        solver = ipxyz(subprob, epsimin=1e-6)
        x_k = solver.update()
        print(solver.iterin)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=solver.w.lam, df=df)

        itte += 1

    logger.info('Alles goed!')


def example_square_mixed(n):

    # Instantiate problem
    prob = Square(n)

    # Define variable and response sets of a mixed approximation scheme as dictionaries
    var_set = {0: np.arange(0, prob.n)}
    resp_set = {0: np.array([0]),
                1: np.array([1])}

    # Instantiate subproblem objects for a mixed approximation scheme
    subprob_map = {
                   (0, 0): Subproblem(intervening=MMA(prob.xmin, prob.xmax),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]])),
                   (1, 0): Subproblem(intervening=Linear(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]]))
                   }

    # Instantiate a mixed scheme
    subprob = Mixed(subprob_map, var_set, resp_set)

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Print current iteration and x_k
        logger.info('iter: {:^4d}  |  x: {:<20s}  |  obj: {:^9.3f}  |  constr: {:^6.3f}'.format(
            itte, np.array2string(x_k[0:2]), f[0], f[1]))
        plotter.plot([f[0], f[1]])

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, prob, subprob, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        itte += 1

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    example_square_Svanberg(2)
    example_square_ipx(2)
    example_square_ipxy(20)
    example_square_ipxyz(20)
    example_square_mixed(20)
