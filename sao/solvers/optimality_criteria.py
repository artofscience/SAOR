import numpy as np


def oc(dg, x=None, gt=0, move=0.2, tol=1e-3, lower=0, upper=1e9):
    x_new = np.zeros_like(x)
    while (upper - lower) / (lower + upper) > tol:
        middle = (lower + upper) / 2
        x_new[:] = x * np.sqrt(-dg[0] / dg[1] / middle)
        x_new[:] = np.clip(np.clip(x_new, x - move, x + move), 0, 1)
        var_change = gt + np.sum((dg[1] * (x_new - x)))
        if var_change > 0:
            lower = middle
        else:
            upper = middle
    return x_new, gt
