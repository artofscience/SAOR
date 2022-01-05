import numpy as np
import logging
from sao.approximations import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.problems.subproblem import Subproblem
from sao.intervening_variables.exponential import Linear
from sao.intervening_variables.mma import MMA02
from sao.intervening_variables.mixed_intervening import MixedIntervening
from sao.move_limits import Bounds, MoveLimit, MoveLimitFraction, AdaptiveMoveLimit
from sao.util import Plot
from sao.solvers.primal_dual_interior_point import pdip, Pdipx, Pdipxyz
from util.plotter import Plot2, Plot3
from Problems._1d.Polynomial_1D import Polynomial1D

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
np.set_printoptions(precision=4)


def example_polynomial_1D():
    logger.info("Solving test_poly using y=MMA and solver=Ipopt Svanberg")

    # Instantiate problem
    problem = Polynomial1D()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2)
    intvar = Linear()     # MMA02(x_min=problem.x_min, x_max=problem.x_max)
    subproblem = Subproblem(SphericalTaylor2(intvar), limits=[bounds, movelimit])

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'criterion', 'max_constr_violation'], path=".")
    plotter2_flag = True
    if plotter2_flag:
        plotter2 = Plot2(problem)

    # Initialize iteration counter and design
    itte = 0
    x_k = problem.x0.copy()

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = problem.g(x_k)
        df = problem.dg(x_k)
        ddf = problem.ddg(x_k) if subproblem.approx.__class__.__name__ == 'Taylor2' else None

        # Build approximate sub-problem at X^(k)
        subproblem.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, problem, subproblem, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k[:] = pdip(subproblem)[0]

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info(
            'iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, np.array2string(x_k[0]), f[0], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_polynomial_1D_mixed():
    logger.info("Solving test_poly using y=MixedMoveLimit and Ipopt Svanberg")

    # Instantiate problem
    problem = Polynomial1D()
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2)
    intvar = MixedIntervening(problem.n, problem.m+1)  # MMA02(x_min=problem.x_min, x_max=problem.x_max)
    intvar.set_intervening(MMA02(x_min=problem.x_min, x_max=problem.x_max), resp=0)
    subproblem = Subproblem(NonSphericalTaylor2(intvar), limits=[bounds, movelimit])

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'criterion', 'max_constr_violation'], path=".")
    plotter3_flag = True
    if plotter3_flag:
        plotter3 = Plot3(problem)

    # Initialize iteration counter and design
    itte = 0
    x_k = problem.x0.copy()

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = problem.g(x_k)
        df = problem.dg(x_k)
        ddf = problem.ddg(x_k) if subproblem.approx.__class__.__name__ == 'Taylor2' else None

        # Build approximate sub-problem at X^(k)
        subproblem.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter3_flag:
            plotter3.plot_pair(x_k, f, problem, subproblem, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k[:] = pdip(subproblem)[0]

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info(
            'iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, np.array2string(x_k[0]), f[0], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    # example_polynomial_1D()
    example_polynomial_1D_mixed()
