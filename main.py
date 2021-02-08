## Imports
import numpy as np
import matplotlib as plt
import constants as ct

from Problems.Mishra_bird import MishraBird
from Problems.Rosenbrock_cubic import RosenCubic
from Problems.Simionescu_func import Simionescu
from Problems.Townsend_func import Townsend
from Problems.Li2015_Fig4 import Li2015Fig4
from Problems.VanderplaatsBeam import Vanderplaats

from Solvers.SolverIP_Svanberg import SvanbergIP

from ConvergenceCriteria.KKT import KKT
from ConvergenceCriteria.ObjChange import ObjectivecChange
from ConvergenceCriteria.VarChange import VariableChange
from ConvergenceCriteria.Alltogether import Alltogether
from ConvergenceCriteria.MaxIteration import MaxIteration

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
    prob = Vanderplaats(100)

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

    # Choose convergence criteria to be used and initialize its object
    criterion = KKT(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = MaxIteration()
    # criterion = ObjectivecChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = VariableChange(xmin=prob.xmin, xmax=prob.xmax)
    # criterion = Alltogether(xmin=prob.xmin, xmax=prob.xmax)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x_init.copy()
    vis = None

    ## OPTIMIZATION LOOP
    while not criterion.converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        g = prob.response(x_k)
        dg = prob.sensitivity(x_k)
        # ddg = prob.sensitivity2(x_k)

        # Print current iteration and x_k
        # print('\titer = {} | X = {} \n'.format(itte, x_k))
        
        # Build approximate sub-problem at X^(k)
        approx.build_sub_prob(x_k, g, dg)
        vis = prob.kout(itte, 0, vis, x_k)
        # approx.build_sub_prob(x_k, g, dg, ddg=ddg)                                                                      # TODO: add the option to have analytical functions as inputs

        # Call solver (x_k, g and dg are within approx instance)
        x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(approx)

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
    main()
    print('\nEnd of optimization ran by main.py\n')





