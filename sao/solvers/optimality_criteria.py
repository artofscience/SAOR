import numpy as np

"""
Nguyen/Paulino OC approach.
"""


def oc2010(problem, x0=None, var_change=0, move=0.2, tol=1e-3, lower=0, upper=1e9):
    x_new = np.zeros_like(x0)
    dg = problem.dg(x0)
    diff = 0
    while (upper - lower) / (lower + upper) > tol:
        middle = (lower + upper) / 2
        x_new[:] = x0 * np.sqrt(-dg[0] / dg[1] / middle)
        x_new[:] = np.clip(np.clip(x_new, x0 - move, x0 + move), 0, 1)
        diff = var_change + np.sum((dg[1] * (x_new - x0)))
        if diff > 0:
            lower = middle
        else:
            upper = middle
    return x_new, diff


"""
OC in line with 99 line Matlab code (Sigmund 1999, Bendsoe 1995).
Knowing that the constraint is a monotonously decreasing function 
of the Lagrang multiplier, the value of the multiplier 
that satisfies the constraint can be found by a bi-sectioning algorithm.

The bi-sectioning algorithm is initialized by guessing a 
lower and upper bound for the Lagrangian multiplier.
The interval which bounds the Lagrange multiplier is repeatedly halved 
until its size is less than the convergence criteria.
"""


def oc1999(problem, x0=None, target=0.5, move=0.2, tol=1e-3, lower=0, upper=1e9):
    x_new = np.zeros_like(x0)
    dg = problem.dg(x0)
    while (upper - lower) / (lower + upper) > tol:
        middle = (lower + upper) / 2
        x_new[:] = x0 * np.sqrt(-dg[0] / dg[1] / middle)
        x_new[:] = np.clip(np.clip(x_new, x0 - move, x0 + move), 0, 1)
        if sum(x_new) > target * np.size(x0):
            lower = middle
        else:
            upper = middle
    return x_new
