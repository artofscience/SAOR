from sao.problems import Subproblem
from sao.approximations import Taylor1
from sao.intervening_variables import MMA
from sao.move_limits import Bounds, MoveLimit, AdaptiveMoveLimit
from sao.convergence_criteria.criteria import VariableChange
from sao.solvers.primal_dual_interior_point import pdip


def mma(problem, x0=None, move=0.2, xmin=0.0, xmax=1.0, asyinit=0.5, asyincr=1.2, asydecr=0.7, asybound=10.0,
        albefa=0.1, oscillation_tol=1e-10, stop_tol=1e-6):
    int_variable = MMA(xmin=xmin, xmax=xmax, asyinit=asyinit, asyincr=asyincr, asydecr=asydecr, asybound=asybound,
                       albefa=albefa, oscillation_tol=oscillation_tol)
    approx = Taylor1(int_variable)
    lim1 = Bounds(problem.xmin, problem.xmax)
    lim2 = MoveLimit(move * (problem.xmax - problem.xmin))
    lim3 = AdaptiveMoveLimit(move_limit=move)
    lims = [lim1, lim2, lim3]

    sub_problem = Subproblem(approx, lims)
    if x0 is None:
        x = problem.x0
    else:
        x = x0
    converged = VariableChange(x, tolerance=stop_tol)

    iter = 0
    while not converged:
        iter += 1
        f = problem.g(x)
        df = problem.dg(x)
        print(iter, ":  ", f[0], x)
        sub_problem.build(x, f, df)
        x[:] = pdip(sub_problem)
    f = problem.g(x)
    return x, f[0]
