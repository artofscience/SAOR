import logging
import numpy as np

from examples.util.plotter_basic import Plot
from examples.util.plotter import Plot2, Plot3
from problems.n_dim.vdp_beam import VanderplaatsBeam

from sao.problems import Subproblem
from sao.approximations import Taylor1
from sao.solvers.primal_dual_interior_point import pdip
from sao.intervening_variables import MixedIntervening
from sao.intervening_variables.mma import MMA02
from sao.move_limits import Bounds, MoveLimitFraction, AdaptiveMoveLimit
from sao.scaling_strategies import InitialObjectiveScaling, InitialResponseScaling

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
np.set_printoptions(precision=4)


def example_vanderplaats(N):
    logger.info("Solving VanderplaatsBeam using y=MMA and solver=pdip")

    # Instantiate problem, intervening variables, approximation, and subproblem
    problem = VanderplaatsBeam(N)
    assert problem.n == 2 * N
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2)
    intvar = MMA02(x_min=problem.x_min, x_max=problem.x_max)
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])

    # Instantiate the scaling strategy
    scaling = InitialResponseScaling(problem.m + 1)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'stress_1', 'tip_disp', 'criterion', 'max_constr_violation'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(problem)

    # Initialize iteration counter and design
    itte = 0
    x_k = problem.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:  # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = problem.g(x_k)
        df = problem.dg(x_k)

        # Apply scaling strategy
        f, df = scaling.scale(f, df)

        # Build approximate sub-problem at X^(k)
        subproblem.build(x_k, f, df)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, problem, subproblem, itte)

        # Solve current subproblem
        x_k = pdip(subproblem)[0]

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = problem.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  stress1: {:^6.3f}  |  tip_disp: {:^6.3f}  |  '
            'criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], f[-1], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], f[-1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_vanderplaats_mixed(N):
    logger.info("Solving VanderplaatsBeam using y=MixedMoveLimit and solver=Ipopt Svanberg")

    # Instantiate problem, intervening variables, approximation, and subproblem
    problem = VanderplaatsBeam(N)
    assert problem.n == 2 * N
    bounds = Bounds(xmin=problem.x_min, xmax=problem.x_max)
    movelimit = MoveLimitFraction(fraction=2)
    intvar = MixedIntervening(problem.n, problem.m + 1)
    intvar.set_intervening(MMA02(x_min=problem.x_min, x_max=problem.x_max), resp=0)
    subproblem = Subproblem(Taylor1(intvar), limits=[bounds, movelimit])

    # Instantiate the scaling strategy
    scaling = InitialObjectiveScaling(problem.m + 1)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'stress_1', 'tip_disp', 'criterion', 'max_constr_violation'], path=".")
    plotter3_flag = False
    if plotter3_flag:
        plotter3 = Plot3(problem)

    # Initialize iteration counter and design
    itte = 0
    x_k = problem.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:  # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = problem.g(x_k)
        df = problem.dg(x_k)

        # Apply scaling strategy
        f, df = scaling.scale(f, df)

        # Build approximate sub-problem at X^(k)
        subproblem.build(x_k, f, df)

        # Plot current approximation
        if plotter3_flag:
            plotter3.plot_pair(x_k, f, problem, subproblem, itte)

        # Solve current subproblem
        x_k = pdip(subproblem)[0]

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = problem.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  stress1: {:^6.3f}  |  tip_disp: {:^6.3f}  |  '
            'criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], f[-1], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], f[-1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    example_vanderplaats(10)
    example_vanderplaats_mixed(10)
