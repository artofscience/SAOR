import numpy as np
import logging
from Problems.Li2015_Fig4 import Li2015Fig4
from sao.approximations.taylor import *
from sao.intervening_variables import *
from sao.move_limits import *
from sao.problems.subproblem import Subproblem
from sao.solvers.SolverIP_Svanberg import SvanbergIP
from sao.solvers.interior_point import InteriorPointXYZ as ipopt
from sao.util import *
from sao.convergence_criteria import *
from sao.scaling_strategies.scaling import *

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

    # Instantiate a mixed move limit strategy
    # mixed_ml = MixedML(prob.n, default=Bound(prob.xmin, prob.xmax))
    # mixed_ml.add_move_limit(MoveLimit(move_limit=0.1), var=[0])
    # mixed_ml.set_move_limit(MoveLimit(move_limit=0.1), var=[0])

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=NonSphericalTaylor2(Linear()))
    subprob.set_limits([Bound(prob.xmin, prob.xmax), MoveLimit(move_limit=0.2)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1', 'constraint_2', f'{criterion.__class__.__name__}',
                    'max_constr_violation'], path=".")
    plotter2_flag = True
    if plotter2_flag:
        plotter2 = Plot2(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Build approximate subproblem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            # plotter2.plot_pair(x_k, f, prob, subprob, itte)
            plotter2.contour_plot(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you choose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & plot g_j and x_i at current iteration
        logger.info('iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  constr1: {:^6.3f}  |'
                    '  constr2: {:^6.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
            itte, np.array2string(x_k[:]), f[0], f[1], f[2], criterion.value, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], f[2], criterion.value, max(0, max(f[1:]))])

        itte += 1

    logger.info('Alles goed!')


def example_truss2d_mixed():
    logger.info("Solving Li2015Fig4 using y=MixedML and Ipopt Svanberg")

    # Instantiate problem
    prob = Li2015Fig4()

    # Instantiate a mixed intervening variable
    mix = Mixed(prob.n, prob.m + 1, default=MMA(prob.xmin, prob.xmax))
    # mix.set_intervening(Linear(), var=[0], resp=[0])
    # mix.set_intervening(Exponential(2), var=[1], resp=[0])      # MMA(prob.xmin, prob.xmax)

    # Instantiate a mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(mix))
    subprob.set_limits([Bound(prob.xmin, prob.xmax), MoveLimit(move_limit=5.0)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1', 'constraint_2', f'{criterion.__class__.__name__}',
                    'max_constr_violation'], path=".")
    plotter3_flag = True
    if plotter3_flag:
        plotter3 = Plot3(prob, responses=np.arange(0, prob.m + 1), variables=np.arange(0, prob.n))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = None

        # Build approximate subproblem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter3_flag:
            # plotter3.plot_pair(x_k, f, prob, subprob, itte)
            plotter3.contour_plot(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & plot g_j and x_i at current iteration
        logger.info('iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  constr1: {:^6.3f}  |'
                    '  constr2: {:^6.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
            itte, np.array2string(x_k[:]), f[0], f[1], f[2], criterion.value, max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], f[2], criterion.value, max(0, max(f[1:]))])

        itte += 1

    logger.info('Alles goed!')


if __name__ == "__main__":
    example_truss2d()
    example_truss2d_mixed()
