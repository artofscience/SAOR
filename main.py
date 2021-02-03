## Imports
import numpy as np
import matplotlib as plt
import constants as ct
from Problems.Mishra_bird import MishraBird
from Problems.Rosenbrock_cubic import RosenCubic
from Problems.Simionescu_func import Simionescu
from Problems.Townsend_func import Townsend
from Problems.Li2015_Fig4 import Li2015Fig4
from Solvers.SolverIP_Svanberg import SvanbergIP
from ConvergenceCriteria.Conv_Criteria import ConvergenceCriteria
from ApproximationSchemes.NonMixed.Lin import Linear
from ApproximationSchemes.NonMixed.CONLIN import CONLIN
from ApproximationSchemes.NonMixed.MMA import MMA
from ApproximationSchemes.Mixed.MixedTemp import MixedTemplate

print('Imports OK')

# In order to use LaTeX format in plots later on
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
np.set_printoptions(precision=4)


## MAIN:  This is the main function used to solve Topology Optimization problems
def main():

    ## INITIALIZATIONS: problem, approximation, solver, convergence criterion

    # Instantiate problem
    prob = Li2015Fig4()

    # # Different variable and response sets (only for a mixed approximation scheme)
    # variable_sets = {0: np.arange(0, 2), 1: np.arange(2, prob.n)}
    # response_sets = {0: np.array([0]), 1: np.array([1]), 2: np.array([2]),
    #                  3: np.array([3]), 4: np.arange(4, 6)}
    #
    # # Instantiating a mixed approximation scheme
    # approx = MixedTemplate(prob.n, prob.m, prob.xmin, prob.xmax,
    #                        approx_array=np.array([['MMA',    'Linear'],
    #                                               ['MMA',    'MMA'   ],
    #                                               ['Linear', 'Linear'],
    #                                               ['Linear', 'MMA'   ],
    #                                               ['MMA',    'MMA'   ]]),
    #                        var_set=variable_sets, resp_set=response_sets)

    # Examples of instantiating non-mixed approximations schemes
    # approx = Linear(prob.n, prob.m, prob.xmin, prob.xmax)
    # approx = CONLIN(prob.n, prob.m, prob.xmin, prob.xmax)
    approx = MMA(prob.n, prob.m, prob.xmin, prob.xmax)

    # Choose solver & initialize its object via Auxiliary_Functions/Choose_Solver.py (no need to do it for each approx)
    solver = SvanbergIP(prob.n, prob.m)

    # Choose convergence criteria to be used and initialize its object
    criterion = ConvergenceCriteria(ct.CRITERION_NAME, prob.xmin, prob.xmax)

    # Initialize iteration counter and design
    itte = 0
    x_k = np.array([2., 1.2])

    ## OPTIMIZATION LOOP
    while (not criterion.converged) and (itte < ct.MAX_ITE_OPT):

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        g = prob.response(x_k)
        dg = prob.sensitivity(x_k)

        # Print current iteration and x_k
        print('\titer = {} | X = {} \n'.format(itte, x_k))
        
        # Build approximate sub-problem at X^(k)
        approx.build_sub_prob(x_k, g, dg)

        # Call solver (x_k, g and dg are within approx instance)
        x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(approx)
        
        # Update old values of approx, current point x_k and increase iteration counter
        itte += 1
        approx.update_old_values(x_k, g, dg, itte)
        x_k = x.copy()

        # Check if convergence criterion is satisfied (give the correct keyword arguments for the criterion you chose)
        criterion.get_Convergence(design=x_k, sensitivities=dg, lagrange_multipliers=lam)

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





