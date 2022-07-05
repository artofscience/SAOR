import numpy as np

from sao.approximations.taylor import Taylor1
from sao.move_limits.move_limit import Bounds
from sao.problems.problem import Problem
from sao.util.tools import parse_to_list


class Subproblem(Problem):
    def __init__(self, functions, limits=Bounds(xmin=0, xmax=1)):
        super().__init__()
        self.functions = functions
        self.set_limits(limits)
        self.lims = parse_to_list(limits)
        self.n, self.m = functions[0].n, len(functions) - 1
        for f in functions:
            if f.n != self.n: print('ERROR')
        self.x_d_k = np.ones(self.m,dtype=float)*0e0 # !!!!!!!!

    def set_limits(self, *limits):
        self.lims = parse_to_list(*limits)

    def add_limits(self, *limits):
        self.lims.extend(parse_to_list(*limits))

    def build(self, x, f, df, ddf=0e0):

        # Update the approximation
#       self.approx.update(x, f, df, ddf)

        # Update the local problem bounds
        self.x_min = np.full_like(x, -np.inf)
        self.x_max = np.full_like(x, +np.inf)

        # Enforce restriction on the possible step size within the subproblem.
        # The step is restricted by the chosen move limit strategy as well as
        # the feasible range of the intervening variables. First the move
        # limits are applied to constraint the step size.
        for ml in self.lims:
            ml.update(x, f, df, ddf)
            ml.clip(self.x_min)
            ml.clip(self.x_max)

        # Additional constraint on the step size by the feasible range of the
        # intervening variables. This prevents the subsolver to make an update
        # that causes the intervening variable to reach unreachable values,
        # e.g. cross the lower/upper bounds in the MMA asymptotes.

        for j in range(self.m+1):
            tmp=self.functions[j].domain()
            for i in range(self.n):
                self.x_min[i]=max(tmp[0][i],self.x_min[i])
                self.x_max[i]=min(tmp[1][i],self.x_max[i])

        assert np.isfinite(self.x_min).all() and np.isfinite(self.x_max).all(), \
            "The bounds must be finite. Use at least one move-limit or bound."

    def g(self, x):
        gg = np.zeros(self.m + 1,dtype=float)
        for j in range(self.m+1):
            f, _, _ = self.functions[j].evaluate(x)
            gg[j] = f
#       print('g',gg)
        return gg

    def dg(self, x):
        dgg = np.zeros((self.m + 1,self.n),dtype=float)
        for j in range(self.m+1):
            _, df, _ = self.functions[j].evaluate(x)
            dgg[j][:] = df
#       print('dg',dgg)
        return dgg

    def ddg(self, x):
        ddgg = np.zeros((self.m +1, self.n)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for j in range(self.m+1):
            _, _, ddf = self.functions[j].evaluate(x)
            ddgg[j][:] = ddf
        return ddgg

    '''
    P = dg_j/dy_ji = dg_j/dx_i * dx_i/dy_ji [(m+1) x n]
    Q = d^2g_j/dy_ji^2 = d^2g_j/dx_i^2 * (dx_i/dy_ji)^2 + dg_j/dx_i * d^2x_i/dy_ji^2 [(m+1) x n]
    y = [(m+1) x n] or [n], depending on the intervening variables used (see reference_files/TaylorExpansion.pdf)
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
