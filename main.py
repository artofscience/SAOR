## Imports
import numpy as np
import matplotlib as plt
import cProfile                                               # to profile the time spent in funcs
import time                                                   # to have better time resolution in profiler

from Problems.Mishra_bird import MishraBird
from Problems.Rosenbrock_cubic import RosenCubic
from Problems.Simionescu_func import Simionescu
from Problems.Townsend_func import Townsend
from Problems.Li2015_Fig4 import Li2015Fig4
from Problems.VanderplaatsBeam import Vanderplaats
from Problems.Top88 import Top88

from Solvers.SolverIP_Svanberg import SvanbergIP

from ConvergenceCriteria.KKT import KKT
from ConvergenceCriteria.Feasibility import Feasibility
from ConvergenceCriteria.ObjChange import ObjectivecChange
from ConvergenceCriteria.VarChange import VariableChange
from ConvergenceCriteria.Alltogether import Alltogether

from ApproximationSchemes.NonMixed.Lin import Linear
from ApproximationSchemes.NonMixed.CONLIN import CONLIN
from ApproximationSchemes.NonMixed.MMA import MMA
from ApproximationSchemes.Mixed.MixedTemp import MixedTemplate

print('Imports OK')

# In order to use LaTeX format in plots later on
# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
np.set_printoptions(precision=4)


## MAIN:  This is the main function used to solve Topology Optimization problems
def main():

    ## INITIALIZATIONS: problem, approximation, solver, convergence criterion

    # Instantiate problem
    prob = Top88(180, 60, 0.4, 3, 5.4, 1)

    # # Instantiating a mixed approximation scheme
    # variable_sets = {0: np.arange(0, 1), 1: np.arange(1, prob.n)}
    # response_sets = {0: np.array([0]), 1: np.array([1]), 2: np.array([2])}
    # approx = MixedTemplate(prob.n, prob.m, prob.xmin, prob.xmax,
    #                        approx_array=np.array([['Linear', 'Linear'],
    #                                               ['MMA'   , 'MMA'   ],
    #                                               ['MMA'   , 'MMA'   ]]),
    #                        var_set=variable_sets, resp_set=response_sets, second_order=True)

    # # Examples of instantiating non-mixed approximations schemes
    # approx = Linear(prob.n, prob.m, prob.xmin, prob.xmax)
    # approx = CONLIN(prob.n, prob.m, prob.xmin, prob.xmax)
    approx = MMA(prob.n, prob.m, prob.xmin, prob.xmax)

    # Choose solver & initialize its object via Auxiliary_Functions/Choose_Solver.py (no need to do it for each approx)
    solver = SvanbergIP(prob.n, prob.m)

    # # Choose convergence criteria to be used and initialize its object
    # criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = ObjectiveChange()
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    vis = None                                  # only for Vanderplaats beam

    ## OPTIMIZATION LOOP
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        g = prob.g(x_k)
        dg = prob.dg(x_k)
        # ddg = prob.ddg(x_k)

        # Print current iteration and x_k
        vis = prob.visualize(vis, x_k, itte)                # visualization of half MBB-beam (99-line code)
        # vis = prob.visualize(itte, 0, vis, x_k)           # visualization of Vanderplaats beam
        print('iter: {:<4d}  |  obj: {:>9.3f}  |  constr: {:>6.3f}  |  vol: {:>6.3f}'.format(
            itte, g[0], g[1], np.mean(np.asarray(prob.H * x_k[np.newaxis].T / prob.Hs)[:, 0])))

        # Build approximate sub-problem at X^(k)
        approx.build_sub_prob(x_k, g, dg)                   # 2nd-order info: approx.build_sub_prob(x_k, g, dg, ddg=ddg)

        # Call solver (x_k, g and dg are within approx instance)
        pr.enable()
        x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(approx)
        pr.disable()

        # Check if convergence criterion is satisfied (give the correct keyword arguments for the criterion you chose)
        criterion.assess_convergence(x_k=x_k, dg=dg, lam=lam, g=g, gold1=approx.gold1, xold1=approx.xold1, iter=itte)

        # Update old values of approx, current point x_k and increase iteration counter
        itte += 1
        approx.update_old_values(x_k, g, dg, itte)
        x_k = x.copy()

    ## RESULTS SECTION
    print('\n\n')
    dash = '-' * 37
    print(dash)
    print('{} solved by {}'.format(prob.name, solver.name))
    print(dash)
    print('{:<20s}{:>4s}'.format('Approx name', 'iterations'))
    print(dash)
    print('{:<20s}{:>4d}'.format(approx.name, itte))


if __name__ == "__main__":

    # Disable profiler cuz you want to measure one thing at a time
    pr = cProfile.Profile(timer=time.perf_counter_ns, timeunit=1e-9, subcalls=True, builtins=True)
    pr.disable()

    main()

    # Print time measurements in the Console
    pr.print_stats(sort=2)

    print('\nEnd of optimization ran by main.py\n')





