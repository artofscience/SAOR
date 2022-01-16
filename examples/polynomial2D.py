import logging
import numpy as np

from problems.two_dim.polynomial_2d import Polynomial2D
from examples.util.plotter import Plot2, Plot3
from examples.util.plotter_basic import Plot

from sao.approximations import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.convergence_criteria import VariableChange
from sao.intervening_variables import Linear, ConLin, MixedIntervening
from sao.intervening_variables.mma import MMA02
from sao.move_limits import Bounds, MoveLimitFraction
from sao.problems import Subproblem
from sao.solvers.primal_dual_interior_point import pdip
from sao.solvers.wrappers.cvxopt import cvxopt_solver
from sao.solvers.wrappers.scipy import scipy_solver


# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
np.set_printoptions(precision=4)


def example_polynomial_2D():
    logger.info("Solving test_poly using y=MMA and solver=pdip")

    # Instantiate problem, intervening variables, approximation, and subproblem
    problem = Polynomial2D()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2)
    intvar = MMA02(x_min=problem.x_min, x_max=problem.x_max)
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'criterion', 'max_constr_violation'], path=".")
    plotter2_flag = True
    if plotter2_flag:
        plotter2 = Plot2(problem)

    # Initialize design and iteration counter
    # x_k = problem.x0.copy()                # At optimum: 1 active constraint (initial design: upper right)
    # x_k = np.array([1.5, 1.6])          # At optimum: 1 active constraint (initial design: lower left)
    # x_k = np.array([1.5, 2.1])          # At optimum: 2 active constraints, i.e. minimum at intersection (upper left)
    x_k = np.array([2, 1.5])  # no constraint active, i.e. internal minimum (lower right)
    itte = 0

    # Optimization loop
    while itte < 100:  # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = problem.g(x_k)
        df = problem.dg(x_k)
        ddf = problem.ddg(x_k) if subproblem.approx.__class__.__name__ == 'Taylor2' else None

        # Build approximate sub-problem at X^(k)
        subproblem.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.contour_plot(x_k, f, problem, subproblem, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k = pdip(subproblem)[0]

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info(
            'iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, np.array2string(x_k[0]), f[0], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_polynomial_2D_mixed():
    logger.info("Solving test_poly using y=MixedML and solver=pdip")

    # Instantiate problem, intervening variables, approximation, and subproblem
    problem = Polynomial2D()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2)
    intvar = MixedIntervening(problem.n, problem.m + 1, default=Linear())
    intvar.set_intervening(MMA02(x_min=problem.x_min, x_max=problem.x_max), resp=0)
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'criterion', 'max_constr_violation'], path=".")
    plotter3_flag = True
    if plotter3_flag:
        plotter3 = Plot3(problem)

    # Initialize design and iteration counter
    # x_k = problem.x0.copy()                # At optimum: 1 active constraint (initial design: upper right)
    # x_k = np.array([1.5, 1.6])             # At optimum: 1 active constraint (initial design: lower left)
    # x_k = np.array([1.5, 2.1])             # At optimum: 2 active constraints (initial design: upper left)
    x_k = np.array([2, 1.5])  # no constraint active, i.e. internal minimum (lower right)
    itte = 0

    # Optimization loop
    while itte < 100:  # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = problem.g(x_k)
        df = problem.dg(x_k)
        ddf = problem.ddg(x_k) if subproblem.approx.__class__.__name__ == 'Taylor2' else None

        # Build approximate sub-problem at X^(k)
        subproblem.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter3_flag:
            # plotter3.plot_pair(x_k, f, prob, subprob, itte)
            plotter3.contour_plot(x_k, f, problem, subproblem, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k = pdip(subproblem)[0]

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info(
            'iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, np.array2string(x_k[0]), f[0], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_polynomial_2D_cvxopt():
    logger.info("Solving test_poly using y=MMA and solver=pdip")

    # Instantiate problem, intervening variables, approximation, and subproblem
    problem = Polynomial2D()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2)
    intvar = MMA02(x_min=problem.x_min, x_max=problem.x_max)
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'criterion', 'max_constr_violation'], path=".")
    plotter2_flag = True
    if plotter2_flag:
        plotter2 = Plot2(problem)

    # Initialize design and iteration counter
    # x_k = problem.x0.copy()                # At optimum: 1 active constraint (initial design: upper right)
    # x_k = np.array([1.5, 1.6])          # At optimum: 1 active constraint (initial design: lower left)
    # x_k = np.array([1.5, 2.1])          # At optimum: 2 active constraints, i.e. minimum at intersection (upper left)
    x_k = np.array([2, 1.5])  # no constraint active, i.e. internal minimum (lower right)
    itte = 0

    # Optimization loop
    while itte < 100:  # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = problem.g(x_k)
        df = problem.dg(x_k)
        ddf = problem.ddg(x_k) if subproblem.approx.__class__.__name__ == 'Taylor2' else None

        # Build approximate sub-problem at X^(k)
        subproblem.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.contour_plot(x_k, f, problem, subproblem, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k = cvxopt_solver(subproblem)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info(
            'iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, np.array2string(x_k[0]), f[0], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_polynomial_2D_scipy():
    logger.info("Solving test_poly using y=MMA and solver=pdip")

    # Instantiate problem, intervening variables, approximation, and subproblem
    problem = Polynomial2D()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2)
    intvar = MMA02(x_min=problem.x_min, x_max=problem.x_max)
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'criterion', 'max_constr_violation'], path=".")
    plotter2_flag = True
    if plotter2_flag:
        plotter2 = Plot2(problem)

    # Initialize design and iteration counter
    # x_k = problem.x0.copy()                # At optimum: 1 active constraint (initial design: upper right)
    # x_k = np.array([1.5, 1.6])          # At optimum: 1 active constraint (initial design: lower left)
    # x_k = np.array([1.5, 2.1])          # At optimum: 2 active constraints, i.e. minimum at intersection (upper left)
    x_k = np.array([2, 1.5])  # no constraint active, i.e. internal minimum (lower right)
    itte = 0

    # Optimization loop
    while itte < 100:  # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = problem.g(x_k)
        df = problem.dg(x_k)
        ddf = problem.ddg(x_k) if subproblem.approx.__class__.__name__ == 'Taylor2' else None

        # Build approximate sub-problem at X^(k)
        subproblem.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.contour_plot(x_k, f, problem, subproblem, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k = scipy_solver(subproblem)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info(
            'iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, np.array2string(x_k[0]), f[0], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    example_polynomial_2D()
    example_polynomial_2D_mixed()
    example_polynomial_2D_cvxopt()
    example_polynomial_2D_scipy()
