import numpy as np
import logging
from Problems.VanderplaatsBeam import Vanderplaats
from sao.approximations.taylor import Taylor1, Taylor2
from sao.intervening_variables import Linear, ConLin, MMA, ReciSquared, ReciCubed
from sao.move_limits.move_limit import MoveLimitIntervening
from sao.problems.subproblem import Subproblem
from sao.problems.mixed import Mixed
from sao.solvers.SolverIP_Svanberg import SvanbergIP
from sao.solvers.interior_point import InteriorPointXYZ as ipopt
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


def example_vanderplaats(N):
    logger.info("Solving Vanderplaats using y=MMA and solver=Ipopt Svanberg")

    # Instantiate problem
    prob = Vanderplaats(N)
    assert prob.n == 2 * N

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1', 'max_constr_violation'], path=".")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None
    solves = 0

    # Optimization loop
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Print current iteration and x_k
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr1: {:^6.3f}  |  constr2: {:^6.3f}  |  '
            'constr3: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], f[1 + N], f[-1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_pair(x_k, f, prob, subprob, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # solver = ipopt(subprob, epsimin=1e-9)
        # x_k = solver.update()
        # solves += solver.iterin

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        itte += 1

    logger.info('Optimization loop converged!')
    print(solves)


def example_vanderplaats_mixed(N):
    logger.info("Solving Vanderplaats using y=MixedScheme and solver=Ipopt Svanberg")

    # Instantiate problem
    prob = Vanderplaats(N)
    assert prob.n == 2 * N

    # Define variable and response sets as dictionaries
    var_set = {0: np.arange(0, N),
               1: np.arange(N, prob.n)}
    resp_set = {0: np.array([0]),
                1: np.arange(1, N + 1),
                2: np.arange(N + 1, prob.n + 1),
                3: np.array([prob.n + 1])}

    # Instantiate a mixed approximation scheme
    subprob_map = {
                   (0, 0): Subproblem(intervening=Linear(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]])),
                   (0, 1): Subproblem(intervening=Linear(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[1]],
                                                              xmax=prob.xmax[var_set[1]])),
                   (1, 0): Subproblem(intervening=ConLin(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]])),
                   (1, 1): Subproblem(intervening=ReciSquared(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[1]],
                                                              xmax=prob.xmax[var_set[1]])),
                   (2, 0): Subproblem(intervening=Linear(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]])),
                   (2, 1): Subproblem(intervening=Linear(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[1]],
                                                              xmax=prob.xmax[var_set[1]])),
                   (3, 0): Subproblem(intervening=ConLin(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[0]],
                                                              xmax=prob.xmax[var_set[0]])),
                   (3, 1): Subproblem(intervening=ReciCubed(),
                                      approximation=Taylor1(),
                                      ml=MoveLimitIntervening(xmin=prob.xmin[var_set[1]],
                                                              xmax=prob.xmax[var_set[1]]))
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
    plotter = Plot(['objective', 'constraint_1', 'max_constr_violation'], path=".")
    plotter3_flag = True
    if plotter3_flag:
        plotter3 = Plot3(prob,
                         responses=np.array([resp_set[i][0] for i in range(len(resp_set.keys()))]),
                         variables=np.arange(0, prob.n))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Print current iteration and x_k
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr1: {:^6.3f}  |  constr2: {:^6.3f}  |  '
            'constr3: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], f[1 + N], f[-1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Plot current approximation
        if plotter3_flag:
            plotter3.plot_pair(x_k, f, prob, subprob, itte)

        # Call solver (x_k, g and dg are within approx instance)
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # solver = ipopt(subprob, epsimin=1e-9)
        # x_k = solver.update()
        # solves += solver.iterin

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        itte += 1

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    # example_vanderplaats(100)
    example_vanderplaats_mixed(3)
