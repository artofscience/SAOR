from sao.problems.problem import Problem
from sao.approximations.taylor import Taylor1
from sao.move_limits.move_limit import MoveLimit
from sao.util.tools import parse_to_list
import numpy as np


class Subproblem(Problem):
    def __init__(self, approximation=Taylor1(), limits=MoveLimit(xmin=0, xmax=1)):
        super().__init__()
        self.approx = approximation
        self.set_limits(limits)
        self.alpha, self.beta = None, None
        self.lims = parse_to_list(limits)

    def set_limits(self, *limits):
        self.lims = parse_to_list(*limits)

    def add_limits(self, *limits):
        self.lims.extend(parse_to_list(*limits))

    def build(self, x, f, df, ddf=None):
        self.n, self.m = len(x), len(f) - 1  # to fit Stijn's solvers # TODO Get from len(x) in the subsolver

        # Update the approximation
        self.approx.update(x, f, df, ddf)

        # Update the local problem bounds
        self.alpha = np.full_like(x, -np.inf)
        self.beta = np.full_like(x, +np.inf)

        # Enforce restriction on the possible step size within the subproblem.
        # The step is restricted by the chosen move limit strategy as well as
        # the feasible range of the intervening variables. First the move
        # limits are applied to constraint the step size.
        for ml in self.lims:
            ml.update(x, f, df, ddf)
            ml.clip(self.alpha)
            ml.clip(self.beta)

        # Additional constraint on the step size by the feasible range of the
        # intervening variables. This prevents the subsolver to make an update
        # that causes the intervening variable to reach unreachable values,
        # e.g. cross the lower/upper bounds in the MMA asymptotes.
        self.approx.clip(self.alpha)
        self.approx.clip(self.beta)

        assert np.isfinite(self.alpha).all() and np.isfinite(self.beta).all(), \
            "The bounds must be finite. Use at least one move-limit or bound."
        # TODO: Possibly a check for finiteness of the bounds

    # TODO These might also be removed if the solver uses prob.approx.g instead of prob.g
    def g(self, x):
        return self.approx.g(x)

    def dg(self, x):
        return self.approx.dg(x)

    def ddg(self, x):
        return self.approx.ddg(x)

    '''
    P = dg_j/dy_ji = dg_j/dx_i * dx_i/dy_ji [(m+1) x n]
    Q = d^2g_j/dy_ji^2 = d^2g_j/dx_i^2 * (dx_i/dy_ji)^2 + dg_j/dx_i * d^2x_i/dy_ji^2 [(m+1) x n]
    y = [(m+1) x n] or [n], depending on the intervening variables used (see ReferenceFiles/TaylorExpansion.pdf)
    x = [n]

          |   resp [1, 2, 3]  |     resp [4, 5]
    -------------------------------------------------
    0-N   |  Taylor1 + MMA    |  Taylor1 + Conlin
    -------------------------------------------------
    N-N+2 |  Taylor1 + linear |  Taylor1 + reciprocal


    P = [P1, P2]
        [P3, P4]

    Q = [0, 0 ]
        [0, Q4]

    '''
