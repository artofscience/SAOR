import numpy as np
from sao.convergence_criteria import VariableChange

"""
OC wrapper.
"""


def oc(problem, x0=None, target=None, move=0.2, tol=1e-3, stop_tol=1e-6):
    x = problem.x0 if x0 is None else x0
    converged = VariableChange(x, tolerance=stop_tol)
    counter = 0
    while not converged:
        counter += 1
        f = problem.g(x)
        print(counter, ":  ", f[0], x)
        x[:] = oc1999(problem, x0=x, target=target, move=move, tol=tol)
    f = problem.g(x)
    return x, f[0]


"""
OC in line with 99 line Matlab code ([Sigmund 1999], [Bendsoe 1995]).

"Knowing that the constraint is a monotonously decreasing function 
of the Lagrang multiplier, the value of the multiplier 
that satisfies the constraint can be found by a bi-sectioning algorithm.

The bi-sectioning algorithm is initialized by guessing a 
lower and upper bound for the Lagrangian multiplier.
The interval which bounds the Lagrange multiplier is repeatedly halved 
until its size is less than the convergence criteria." [Sigmund 1999]

Small modifications are made if target is not provided (target=None).
In that case the update ensures the material usage stays constant
"""


def oc1999(problem, x0=None, target=None, move=0.2, tol=1e-3, lower=0, upper=1e9):
    x_new = problem.x0 if x0 is None else x0
    target = np.sum(x_new) if target is None else target  # target material usage
    dg = problem.dg(x0)  # get sensitivities from (sub)problem
    while (upper - lower) / (lower + upper) > tol:  # loop until Lagrange multiplier is found (within tolerance)
        middle = (lower + upper) / 2
        x_new[:] = x0 * np.sqrt(-dg[0] / dg[1] / middle)  # set step in direction of objective sensitivities
        x_new[:] = np.clip(np.clip(x_new, x0 - move, x0 + move), 0, 1)  # clip by move limit
        if np.sum(x_new) - target > 0:  # if material usage is too large increase multiplier else decrease
            lower = middle
        else:
            upper = middle
    return x_new
