import logging

import numpy as np

from examples.util.plotter import Plot2, Plot3
from problems.two_dim.li2015_fig4 import Li2015Fig4
from sao.approximations import Taylor1, SphericalTaylor2
from sao.intervening_variables import Linear, Exponential, MixedIntervening
from sao.intervening_variables.mma import MMA02 as MMA
from sao.move_limits import Bounds, MoveLimit
from sao.problems import Subproblem
from sao.solvers.pdip_svanberg import ipsolver
from sao.util import Plot

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
    subprob = Subproblem(approximation=Taylor1(MMA(prob.x_min, prob.x_max)))
    subprob.set_limits([Bounds(prob.x_min, prob.x_max), MoveLimit(move_limit=5.0)])

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint_1', 'constraint_2', 'criterion',
                    'max_constr_violation'], path=".")
    plotter2_flag = True
    if plotter2_flag:
        plotter2 = Plot2(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while itte < 100:  # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = prob.ddg(x_k) if isinstance(subprob.approx, Taylor1) else None

        # Build approximate subproblem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            # plotter2.plot_pair(x_k, f, prob, subprob, itte)
            plotter2.contour_plot(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k = ipsolver(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you choose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info('iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  constr1: {:^6.3f}  |'
                    '  constr2: {:^6.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
            itte, np.array2string(x_k[:]), f[0], f[1], f[2], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], f[2], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Alles goed!')


def example_truss2d_mixed():
    logger.info("Solving Li2015Fig4 using y=MixedMoveLimit and Ipopt Svanberg")

    # Instantiate problem
    prob = Li2015Fig4()

    # Instantiate a mixed intervening variable
    mix = MixedIntervening(prob.n, prob.m + 1, default=MMA(prob.x_min, prob.x_max))
    mix.set_intervening(Linear(), var=[0], resp=[0])
    mix.set_intervening(Exponential(2), var=[1], resp=[0])  # MMA(prob.x_min, prob.x_max)

    # Instantiate a mixed approximation scheme
    subprob = Subproblem(approximation=SphericalTaylor2(mix))
    subprob.set_limits([Bounds(prob.x_min, prob.x_max), MoveLimit(move_limit=5.0)])

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint_1', 'constraint_2', 'criterion',
                    'max_constr_violation'], path=".")
    plotter3_flag = True
    if plotter3_flag:
        plotter3 = Plot3(prob, responses=np.arange(0, prob.m + 1), variables=np.arange(0, prob.n))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while itte < 100:  # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = prob.ddg(x_k) if isinstance(subprob.approx, Taylor1) else None

        # Build approximate subproblem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter3_flag:
            # plotter3.plot_pair(x_k, f, prob, subprob, itte)
            plotter3.contour_plot(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k = ipsolver(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot
        logger.info('iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  constr1: {:^6.3f}  |'
                    '  constr2: {:^6.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
            itte, np.array2string(x_k[:]), f[0], f[1], f[2], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], f[2], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Alles goed!')


if __name__ == "__main__":
    example_truss2d()
    example_truss2d_mixed()
