import numpy as np
import logging
from sao.approximations import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.problems import Problem, Subproblem
from sao.intervening_variables import Linear, MMA, MMAsquared, MixedIntervening
from sao.move_limits import Bounds, MoveLimit, AdaptiveMoveLimit
from sao.convergence_criteria import ObjectiveChange, VariableChange, IterationCount, Feasibility
from sao.scaling_strategies import InitialObjectiveScaling, InitialResponseScaling
from sao.util import Plot
from sao.solvers import SvanbergIP
from util.plotter import Plot2, Plot3
from Problems.VanderplaatsBeam import Vanderplaats

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
    logger.info("Solving Vanderplaats using y=MMA and solver=Ipopt Svanberg")

    # Instantiate problem
    prob = Vanderplaats(N)
    assert prob.n == 2 * N

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(MMA(prob.xmin, prob.xmax)))
    subprob.set_limits([Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=5.0)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate the scaling strategy
    scaling = InitialResponseScaling(prob.m+1)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'stress_1', 'tip_disp', 'criterion', 'max_constr_violation'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Apply scaling strategy
        f, df = scaling.scale(f, df)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you choose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  stress1: {:^6.3f}  |  tip_disp: {:^6.3f}  |  '
            'criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], f[-1], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], f[-1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_vanderplaats_mixed(N):
    logger.info("Solving Vanderplaats using y=MixedMoveLimit and solver=Ipopt Svanberg")

    # Instantiate problem
    prob = Vanderplaats(N)
    assert prob.n == 2 * N

    # Instantiate a mixed intervening variable
    mix = MixedIntervening(prob.n, prob.m + 1, default=MMAsquared(prob.xmin, prob.xmax))
    # mix.set_intervening(ReciCubed(), var=np.arange(0, N), resp=np.arange(1, N + 1))
    # mix.set_intervening(ReciCubed(), var=np.arange(N, prob.n), resp=np.arange(1, N + 1))
    # mix.set_intervening(ReciCubed(), var=np.arange(0, N), resp=[prob.m])
    # mix.set_intervening(ReciCubed(), var=np.arange(N, prob.n), resp=[prob.m])
    # mix.set_intervening(Linear(), var=np.arange(0, prob.n) , resp=np.arange(N+1, prob.m))

    # Instantiate a mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(mix))
    subprob.set_limits([Bounds(prob.xmin, prob.xmax), AdaptiveMoveLimit(move_limit=5.0)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate the scaling strategy
    scaling = InitialObjectiveScaling(prob.m + 1)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'stress_1', 'tip_disp', 'criterion', 'max_constr_violation'], path=".")
    plotter3_flag = False
    if plotter3_flag:
        plotter3 = Plot3(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:        # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Apply scaling strategy
        f, df = scaling.scale(f, df)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Plot current approximation
        if plotter3_flag:
            plotter3.plot_pair(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you choose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = prob.visualize(x_k, itte, vis)
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
