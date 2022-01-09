import numpy as np

from problems.topology_optimization import utils
from sao.convergence_criteria import IterationCount
from sao.solvers.primal_dual_interior_point import pdip


def optimize(problem, sub_problem, x, itercount):
    converged = IterationCount(itercount)
    counter = 0
    output = np.zeros((itercount - 1, 5), dtype=float)
    output2 = np.zeros((itercount - 1, 5), dtype=float)
    # setup for plotting design
    plotter = utils.PlotDesign(problem, x)

    f = [0, 0]
    xold = np.zeros_like(x)
    change = np.zeros_like(x)
    expected_lin = 1
    expected_approx = 1

    while not converged:
        plotter.plot(x, counter)

        fold = f[0]
        ## RESPONSES
        f = problem.g(x)
        df = problem.dg(x)
        ##

        # # Scaling
        # if counter == 0:
        #     problem.factor = f[0]
        # f[0] = f[0] / problem.factor
        # df[0, :] = df[0, :] / problem.factor

        print(counter, ":  ", f[0])
        output[counter, 0:2] = f[:]  # objective and constraint values
        output[counter, 2] = np.abs(f[0] - fold)  # objective change
        output2[counter, 0] = (f[0] - fold) / expected_lin  # relative objective change
        output2[counter, 1] = (f[0] - fold) / expected_approx

        xphys = problem.filter.forward(x)
        output2[counter, 2] = np.sum(4 * xphys.flatten() * (1 - xphys.flatten()) / problem.mesh.n)

        ## BUILD SUBPROBLEM
        sub_problem.build(x, f, df)
        ##

        xold[:] = x[:]
        ## SOLVE SUBPROBLEM
        x[:], no_inner_iter = pdip(sub_problem)
        ##
        output2[counter, 3] = no_inner_iter  # number of inner iterations in pdip

        changeold = change
        change = x - xold
        expected_lin = np.dot(df[0, :], change)
        expected_approx = sub_problem.g(x)[0] - sub_problem.g(xold)[0]

        output[counter, 3] = np.mean(np.abs(change))  # mean variable change
        output[counter, 4] = np.max(np.abs(change))  # max variable change
        output2[counter, 4] = np.dot(changeold, change) / \
                              np.dot(np.linalg.norm(changeold), np.linalg.norm(change))

        counter += 1

    output[0, [2, 3, 4]] = np.nan
    output2[0, [0, 1, 4]] = np.nan

    return output, output2, xphys
