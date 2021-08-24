import numpy as np
import logging
from sao.approximations import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.problems import Problem, Subproblem
from sao.intervening_variables import Linear, MMA, MixedIntervening
from sao.move_limits import Bounds, MoveLimit, AdaptiveMoveLimit
from sao.convergence_criteria import ObjectiveChange, VariableChange, IterationCount, Feasibility
from sao.util import Plot
from sao.solvers import SvanbergIP
from util.plotter import Plot2, Plot3
from Problems.Square import Square

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
np.set_printoptions(precision=4)


def example_square(n):
    logger.info("Solving Square.py using y=MMA and solver=SvanbergIP")

    # Instantiate problem
    prob = Square(n)

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(MMA(prob.xmin, prob.xmax)))
    subprob.set_limits([Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.1)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(x_min=prob.x_min, x_max=prob.x_max)
    # converged = ObjectiveChange(prob)
    # converged = Feasibility(prob)

    # converged = ObjectiveChange(prob.f[0]) & Feasibility(prob.f[1:], slack=1e-3) | IterationCount(5)

    # criterion = ObjectiveChange(prob.f[0])
    # criterion = VariableChange(x_min=prob.x_min, x_max=prob.x_max)
    # criterion = Feasibility()
    # criterion = Alltogether(x_min=prob.x_min, x_max=prob.x_max)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'criterion', 'max_constr_violation'], path=".")
    plotter2_flag = True
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0, 1]), variables=np.arange(0, prob.n))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while itte < 100:       # not converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion(x_k=x_k, obj=f[0], constraints=f[1:], iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info('iter: {:^4d}  |  x: {:<20s}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  criterion: {:^6.3f}  '
                    '|  max_constr_viol: {:^6.3f}'.format(itte, np.array2string(x_k[0:2]), f[0], f[1],
                                                          0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_square_mixed(n):
    logger.info("Solving Square.py using y=MixedMoveLimit and solver=SvanbergIP")

    # Instantiate problem
    prob = Square(n)

    # Instantiate a mixed intervening variable
    mix = MixedIntervening(prob.n, prob.m + 1, default=Linear())
    mix.set_intervening(MMA(prob.xmin, prob.xmax), var=[0], resp=[1])

    # Instantiate a mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(mix))
    subprob.set_limits([Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.5)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(x_min=prob.x_min, x_max=prob.x_max)
    # criterion = ObjectiveChange(prob.f)
    # criterion = VariableChange(x_min=prob.x_min, x_max=prob.x_max)
    # criterion = Feasibility()
    # criterion = Alltogether(x_min=prob.x_min, x_max=prob.x_max)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'criterion', 'max_constr_violation'], path=".")
    plotter3_flag = True
    if plotter3_flag:
        plotter3 = Plot3(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter3_flag:
            plotter3.plot_pair(x_k, f, prob, subprob, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info(
            'iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, np.array2string(x_k[0]), f[0], 0, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    example_square(2)
    example_square_mixed(2)

