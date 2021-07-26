import numpy as np
import logging
from sao.approximations import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.problems import Problem, Subproblem
from sao.intervening_variables import Linear, MMA, MMASquared, MixedIntervening
from sao.move_limits import Bounds, MoveLimit, AdaptiveMoveLimit
from sao.convergence_criteria import ObjectiveChange, VariableChange, IterationCount, Feasibility
from sao.scaling_strategies import InitialObjectiveScaling, InitialResponseScaling
from sao.util import Plot
from sao.solvers import SvanbergIP
from util.plotter import Plot2, Plot3
from Problems.topology_optimization_benchmark.compliance import Compliance
from Problems.topology_optimization_benchmark.stress import Stress
from Problems.topology_optimization_benchmark.mechanism import Mechanism
from Problems.topology_optimization_benchmark.eigenvalue import Eigenvalue
from Problems.topology_optimization_benchmark.dynamic_compliance import DynamicCompliance
from Problems.topology_optimization_benchmark.selfweight import Selfweight
from Problems.topology_optimization_benchmark.thermomech import Thermomech

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
np.set_printoptions(precision=4)

# # If you want to write to a .log file (stored in the same directory as the script you run)
# file_handler = logging.FileHandler('test_mixed_square.log')
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


def example_compliance(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3):
    logger.info("Solving compliance minimization subject to volume constraint")

    # Instantiate problem
    prob = Compliance(nelx, nely, volfrac, penal, rmin)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(intervening=MMA(prob.xmin, prob.xmax, asydecr=0.6, asyincr=1.3)),
                         limits=[Bounds(xmin=0., xmax=1.), MoveLimit(move_limit=0.1)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0]), variables=np.arange(3, prob.n, 100))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_dynamic_compliance(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3):
    logger.info("Solving dynamic compliance minimization subject to volume constraint")

    # Instantiate problem
    prob = DynamicCompliance(nelx, nely, volfrac, penal, rmin, omega=2500, plot_frf=True)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(intervening=MMA(prob.xmin, prob.xmax)),
                         limits=[Bounds(xmin=0, xmax=1), MoveLimit(move_limit=0.1)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([0]), variables=np.arange(3, prob.n, 100))

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_stress(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3, max_stress=1):
    logger.info("Solving volume minimization subject to aggregated stress constraint")

    # Instantiate problem
    prob = Stress(nelx, nely, volfrac, penal, rmin, max_stress=max_stress)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(intervening=MMA(prob.xmin, prob.xmax, asyincr=1.3, asydecr=0.6)),
                         limits=[Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.1)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([1]), variables=np.arange(3, prob.n, 100))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


# TODO: uniform initial design ends at a local minimum where g_0 = 0.0
def example_mechanism(nelx=100, nely=50, volfrac=0.3, penal=3, rmin=3, kin=0.01, kout=0.01):
    logger.info("Solving displacement minimization subject to volume constraint")

    # Instantiate problem
    prob = Mechanism(nelx, nely, volfrac, penal, rmin, kin=kin, kout=kout)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(intervening=MMA(prob.xmin, prob.xmax)),
                         limits=[Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.3)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([1]), variables=np.arange(3, prob.n, 100))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_eigenvalue(nelx=100, nely=50, volfrac=0.6, penal=3, rmin=3):
    logger.info("Solving eigenfrequency maximization subject to volume constraint")

    # Instantiate problem
    prob = Eigenvalue(nelx, nely, volfrac, penal, rmin)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(intervening=MMA(prob.xmin, prob.xmax)),
                         limits=[Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.3)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([1]), variables=np.arange(3, prob.n, 100))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_self_weight(nelx=100, nely=50, volfrac=0.1, penal=3, rmin=3, load=1.0, gravity=1.0):
    logger.info("Solving compliance minimization subject to volume constraint")

    # Instantiate problem
    prob = Selfweight(nelx, nely, volfrac, penal, rmin, load, gravity)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(intervening=MMA(prob.xmin, prob.xmax)),
                         limits=[Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.3)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([1]), variables=np.arange(3, prob.n, 100))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_thermomech(nelx=100, nely=50, volfrac=0.1, penal=3, rmin=3, load=1.0, gravity=1.0):
    logger.info("Solving compliance minimization subject to volume constraint")

    # Instantiate problem
    prob = Thermomech(nelx, nely, volfrac, penal, rmin, load, gravity)
    assert prob.n == nelx * nely

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(intervening=MMA(prob.xmin, prob.xmax)),
                         limits=[Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.3)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter2_flag = False
    if plotter2_flag:
        plotter2 = Plot2(prob, responses=np.array([1]), variables=np.arange(3, prob.n, 100))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        # Assess convergence (give the correct keyword arguments for the criterion you chose)
        # criterion.assess_convergence(x_k=x_k, f=f, iter=itte, lam=lam, df=df)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_compliance_mixed(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3):
    logger.info("Solving compliance minimization subject to volume constraint with y=MixedIntervening")

    # Instantiate problem
    prob = Compliance(nelx, nely, volfrac, penal, rmin)
    assert prob.n == nelx * nely

    # Instantiate a mixed intervening variable
    mix = MixedIntervening(prob.n, prob.m + 1, default=MMA(prob.xmin, prob.xmax))
    mix.set_intervening(Linear(), var=np.arange(0, prob.n), resp=[1])

    # Instantiate a mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(mix))
    subprob.set_limits([Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.1)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter3_flag = False
    if plotter3_flag:
        plotter3 = Plot3(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

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
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_stress_mixed(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=2, max_stress=1):
    logger.info("Solving weight minimization subject to aggregated stress constraint with y=MixedIntervening")

    # Instantiate problem
    prob = Stress(nelx, nely, volfrac, penal, rmin, max_stress=max_stress)
    assert prob.n == nelx * nely

    # Instantiate a mixed intervening variable
    mix = MixedIntervening(prob.n, prob.m + 1, default=MMA(prob.xmin, prob.xmax))
    mix.set_intervening(Linear(), var=np.arange(0, prob.n), resp=[0])

    # Instantiate a mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(mix))
    subprob.set_limits([Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.1)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, x_max=prob.x_max)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, x_max=prob.x_max)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter3_flag = False
    if plotter3_flag:
        plotter3 = Plot3(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

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
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_mechanism_mixed(nelx=100, nely=50, volfrac=0.3, penal=3, rmin=3, kin=0.01, kout=0.001):
    logger.info("Solving displacement minimization subject to volume constraint")

    # Instantiate problem
    prob = Mechanism(nelx, nely, volfrac, penal, rmin, kin=kin, kout=kout)
    assert prob.n == nelx * nely

    # Instantiate a mixed intervening variable
    mix = MixedIntervening(prob.n, prob.m + 1, default=MMA(prob.xmin, prob.xmax))
    mix.set_intervening(Linear(), var=np.arange(0, prob.n), resp=[1])

    # Instantiate a mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(mix))
    subprob.set_limits([Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.1)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, x_max=prob.x_max)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, x_max=prob.x_max)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter3_flag = False
    if plotter3_flag:
        plotter3 = Plot3(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

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
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


def example_eigenvalue_mixed(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3):
    logger.info("Solving eigenfrequency maximization subject to volume constraint with y=MixedIntervening")

    # Instantiate problem
    prob = Eigenvalue(nelx, nely, volfrac, penal, rmin)
    assert prob.n == nelx * nely

    # Instantiate a mixed intervening variable
    mix = MixedIntervening(prob.n, prob.m + 1, default=MMA(prob.xmin, prob.xmax))
    mix.set_intervening(Linear(), var=np.arange(0, prob.n), resp=[1])

    # Instantiate a mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(mix))
    subprob.set_limits([Bounds(prob.xmin, prob.xmax), MoveLimit(move_limit=0.1)])

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate convergence criterion
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Feasibility()
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Instantiate plotter           # TODO: Change the 'criterion' to f'{criterion.__class__.__name__}'
    plotter = Plot(['objective', 'constraint', 'max_constr_violation'], path="../../../../Desktop")
    plotter3_flag = False
    if plotter3_flag:
        plotter3 = Plot3(prob)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None

    # Optimization loop
    while itte < 100:       # not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

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
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, f[0], f[1], max(0, max(f[1:]))))
        plotter.plot([f[0], f[1], max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    # TODO: Create the `prob` object here and call a function `optimize(prob) to improve running these examples`
    # Non-mixed optimizers (use nelx=50, nely=20 for plotter2)
    # example_compliance(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3)
    # example_dynamic_compliance(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3)
    example_stress(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3, max_stress=1)
    example_mechanism(nelx=100, nely=50, volfrac=0.3, penal=3, rmin=3, kin=0.001, kout=0.0001)
    example_eigenvalue(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3)
    example_thermomech(nelx=200, nely=200, volfrac=0.3, penal=3, rmin=3, load=-0.1, gravity=100)

    # Mixed optimizers
    example_compliance_mixed(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3)
    example_stress_mixed(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3, max_stress=1)
    example_mechanism_mixed(nelx=100, nely=50, volfrac=0.3, penal=3, rmin=3, kin=0.001, kout=0.0001)
    example_eigenvalue_mixed(nelx=100, nely=50, volfrac=0.4, penal=3, rmin=3)
