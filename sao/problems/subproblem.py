import numpy as np

from sao.problems.problem import Problem
from sao.intervening_variables import Linear
from sao.approximations.taylor import Taylor1
from sao.move_limits.move_limit import MoveLimit


class Subproblem(Problem):
    def __init__(self, intervening=Linear(), approximation=Taylor1(), ml=MoveLimit()):
        super().__init__()
        self.inter = intervening
        self.approx = approximation
        self.ml = ml
        self.alpha, self.beta = None, None

    def build(self, x, f, df, ddf=None):
        self.n, self.m = len(x), len(f) - 1             # to fit Stijn's solvers

        self.inter.update(x, f, df)

        # If available, handle 2nd-order information
        if ddf is not None:
            self.approx.update(x, self.inter.y, f, df, self.inter.dxdy, ddf=ddf, ddxddy=self.inter.ddxddy)
        else:
            self.approx.update(x, self.inter.y, f, df, self.inter.dxdy)

        # Enforce restriction on the possible step size within the subproblem.
        # The step is restricted by the chosen move limit strategy as well as
        # the feasible range of the intervening variables. First the move
        # limits are applied to constraint the step size.
        self.alpha, self.beta = self.ml(x)

        # Additional constraint on the step size by the feasible range of the
        # intervening variables. This prevents the subsolver to make an update
        # that causes the intervening variable to reach unreachable values,
        # e.g. cross the lower/upper bounds in the MMA asymptotes.
        self.alpha = np.maximum(self.alpha, self.inter.alpha)
        self.beta = np.minimum(self.beta, self.inter.beta)

    def g(self, x):
        return self.approx.g(self.inter.y(x).T)

    def dg(self, x):
        return self.approx.dg(self.inter.y(x).T, self.inter.dydx(x))

    def ddg(self, x):
        return self.approx.ddg(self.inter.y(x).T, self.inter.dydx(x), self.inter.ddyddx(x))

    def g_dg(self, x):
        # save repeatedly used computations
        y = self.inter.y(x).T
        dydx = self.inter.dydx(x)
        return self.approx.g(y), self.approx.dg(y, dydx)

    def g_dg_ddg(self, x):
        # save repeatedly used computations
        y = self.inter.y(x).T
        dydx = self.inter.dydx(x)

        g = self.approx.g(y)
        dg = self.approx.dg(y, dydx)
        ddg = self.approx.ddg(y, dydx, self.inter.ddyddx(x))
        return g, dg, ddg

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
