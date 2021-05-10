import numpy as np
import logging
from Problems.Li2015_Fig4 import Li2015Fig4
from sao.approximations.taylor import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.intervening_variables import Linear, ConLin, MMA
from sao.move_limits.move_limit import MoveLimitIntervening, MoveLimitMMA
from sao.problems.subproblem import Subproblem
from sao.problems.mixed import Mixed
from sao.solvers.SolverIP_Svanberg import SvanbergIP
from sao.util.plotter import Plot, Plot2, Plot3
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


def example_truss2d():
    logger.info("Solving Li2015Fig4 using Ipopt Svanberg")

    # Instantiate problem
    prob = Li2015Fig4()

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(force_convex=True),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax, move_limit=15.0))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1', 'constraint_2'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob)

    # Optimization loop
    while itte < 50:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Print & plot g_j and x_i at current iteration
        logger.info('iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  constr1: {:^6.3f}  |  constr2: {:^6.3f}'.format(
            itte, np.array2string(x_k[:]), f[0], f[1], f[2]))
        plotter.plot([f[0], f[1], f[2]])

        # Build approximate subproblem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        itte += 1

    logger.info('Alles goed!')


def example_truss2d_mixed():
    logger.info("Solving Li2015Fig4 using Ipopt Svanberg")

    # Instantiate problem
    prob = Li2015Fig4()

    # Define variable and response sets as dictionaries
    var_set = {0: np.array([0]),
               1: np.array([1])}
    resp_set = {0: np.array([0]),
                1: np.array([1, 2])}

    # Instantiate a mixed approximation scheme
    subprob_map = {
                   (0, 0): Subproblem(intervening=Linear(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]],
                                                              move_limit=15.0)),
                   (0, 1): Subproblem(intervening=Linear(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[1]],
                                                              xmax=prob.xmax[var_set[1]],
                                                              move_limit=15.0)),
                   (1, 0): Subproblem(intervening=MMA(prob.xmin[var_set[0]], prob.xmax[var_set[0]]),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]],
                                                              move_limit=15.0)),
                   (1, 1): Subproblem(intervening=MMA(prob.xmin[var_set[1]], prob.xmax[var_set[1]]),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[1]],
                                                              xmax=prob.xmax[var_set[1]],
                                                              move_limit=15.0)),
                   # (2, 0): Subproblem(intervening=Linear(),
                   #                    approximation=Taylor1(),
                   #                    ml=MoveLimitMMA(xmin=prob.xmin[var_set[0]],
                   #                                    xmax=prob.xmax[var_set[0]],
                   #                                    move_limit=15.0)),
                   # (2, 1): Subproblem(intervening=ConLin(),
                   #                    approximation=Taylor1(),
                   #                    ml=MoveLimitMMA(xmin=prob.xmin[var_set[1]],
                   #                                    xmax=prob.xmax[var_set[1]],
                   #                                    move_limit=15.0))
                   }

    # Instantiate a mixed scheme
    subprob = Mixed(subprob_map, var_set, resp_set)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1', 'constraint_2'], path=".")
    plotter3_flag = True
    if plotter3_flag:
        plotter3 = Plot3(prob, responses=np.arange(0, prob.m + 1), variables=np.arange(0, prob.n))

    # Optimization loop
    # while itte < 500:
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = None

        # Print & plot g_j and x_i at current iteration
        logger.info('iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}'.format(itte, np.array2string(x_k[0]), f[0]))
        plotter.plot([f[0], f[1], f[2]])

        # Build approximate subproblem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter3_flag:
            plotter3.plot_pair(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        itte += 1

    logger.info('Alles goed!')


if __name__ == "__main__":
    # example_truss2d()
    example_truss2d_mixed()
