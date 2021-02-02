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

    # Initialize problem
    prob = Li2015Fig4()

    # # Initialize different variable and response sets for a mixed approximation scheme
    # variable_sets = {0: np.arange(0, 2), 1: np.arange(2, prob.n)}
    # response_sets = {0: np.array([0]), 1: np.array([1]), 2: np.array([2]),
    #                  3: np.array([3]), 4: np.arange(4, 6)}
    # variable_sets = {0: np.arange(0, prob.n)}
    # response_sets = {0: np.arange(0, prob.m + 1)}
    #
    # Initialize approx object
    # approx = MixedTemplate(prob.n, prob.m, prob.xmin, prob.xmax,
    #                        approx_array=np.array([['MMA',    'Linear'],
    #                                               ['MMA',    'MMA'   ],
    #                                               ['Linear', 'Linear'],
    #                                               ['Linear', 'MMA'   ],
    #                                               ['MMA',    'MMA'   ]]),
    #                        var_set=variable_sets, resp_set=response_sets)
    # approx = MixedTemplate(prob.n, prob.m, prob.xmin, prob.xmax,
    #                        approx_array=np.array([['MMA']]),
    #                        var_set=variable_sets, resp_set=response_sets)
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

        # Build approximate sub-problem at X^(k)
        approx.build_sub_prob(x_k, g, dg)

        # Call solver (x_k, g and dg are within approx instance)
        x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(approx)

        # Update old x_k, g and dg values
        approx.update_old_values(x_k, g, dg, itte)

        # Rerun problem for new point found by solver (i.e. x)
        x_k = x.copy()

        # Increase counters for optimization loop
        itte += 1
        approx.iter += 1
        # for j in range(0, approx.num_of_resp_sets):
        #     for i in range(0, approx.num_of_var_sets):
        #         approx.approx_obj[j, i].iter = approx.iter

        # Calculate all convergence criteria & check for convergence
        KKT_res = criterion.get_KKT_norm(x_k, dg, lam)
        VarChange_norm = criterion.get_VarChange(x_k, approx.xold1)
        ObjChange_abs = criterion.get_ObjChange(g[0], approx.gold1[0])
        if (criterion.name == 'KKT') and (KKT_res < ct.TOLERANCE):
            criterion.converged = True
        elif (criterion.name == 'VariableChange') and (VarChange_norm < ct.TOLERANCE) and \
             (np.all(g[1:] < ct.TOLERANCE)):
            criterion.converged = True
        elif (criterion.name == 'ObjectiveChange') and (ObjChange_abs < ct.TOLERANCE) and \
             (np.all(g[1:] < ct.TOLERANCE)):
            criterion.converged = True
        elif (criterion.name == 'AllTogether') and (KKT_res < ct.TOLERANCE) and \
             (VarChange_norm < ct.TOLERANCE) and (ObjChange_abs < ct.TOLERANCE) and \
             (np.all(g[1:] < ct.TOLERANCE)):
            criterion.converged = True

        # Print current iteration and x_k
        print('\titer = {} | X = {} \n'.format(itte, x_k))

    ## RESULTS
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





