from sao.approximations import Taylor1
from sao.convergence_criteria.change import VariableChange
from sao.intervening_variables.mma import MMA02 as MMA
from sao.move_limits import Bounds, MoveLimit, AdaptiveMoveLimit
from sao.problems import Subproblem
from sao.solvers.primal_dual_interior_point import pdip


def mma(problem, x0=None, move=0.2, xmin=0.0, xmax=1.0, stop_tol=1e-6):
    int_variable = MMA(x_min=xmin, x_max=xmax)
    approx = Taylor1(int_variable)
    lim1 = Bounds(problem.x_min, problem.x_max)
    lim2 = MoveLimit(move * (problem.x_max - problem.x_min))
    lim3 = AdaptiveMoveLimit(move_limit=move)
    lims = [lim1, lim2, lim3]

    sub_problem = Subproblem(approx, lims)
    x = problem.x0 if x0 is None else x0
    converged = VariableChange(x, tolerance=stop_tol)

    iter = 0
    while not converged:
        iter += 1
        f = problem.g(x)
        df = problem.dg(x)
        print(iter, ":  ", f[0], x)
        sub_problem.build(x, f, df)
        x[:] = pdip(sub_problem)[0]
    f = problem.g(x)
    return x, f[0]
