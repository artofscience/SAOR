import numpy as np

from sao.approximations.taylor import Taylor1
from sao.move_limits.move_limit import Bounds
from sao.problems.subproblem import Subproblem as sub
from sao.util.tools import parse_to_list


class Subproblem(sub):
    def __init__(self, functions, limits=Bounds(xmin=0, xmax=1)):
        super().__init__()
        self.functions = functions
        self.set_limits(limits)
        self.lims = parse_to_list(limits)
        self.n, self.m = functions[0].n, len(functions) - 1
        for f in functions:
            if f.n != self.n: print('ERROR')
        self.x_d_k = np.ones(self.m,dtype=float)*0e0 # !!!!!!!!
#
    def build(self, x, f, df, ddf=0e0):

        # Update the approximation
#       self.approx.update(x, f, df, ddf)

        self.x_k = x 

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
            self.x_min=np.maximum(tmp[0],self.x_min)
            self.x_max=np.minimum(tmp[1],self.x_max)

        assert np.isfinite(self.x_min).all() and np.isfinite(self.x_max).all(), \
            "The bounds must be finite. Use at least one move-limit or bound."

    def g(self, x):
        _g = np.zeros(self.m + 1,dtype=float)
        for j in range(self.m+1):
            _g[j] = self.functions[j].g(x)
        return _g

    def dg(self, x):
        _dg = np.zeros((self.m + 1,self.n),dtype=float)
        for j in range(self.m+1):
            _dg[j][:] = self.functions[j].dg(x)
        return _dg

    def ddg(self, x):
        _dgg = np.zeros((self.m +1, self.n)) 
        for j in range(self.m+1):
            _dgg[j][:] = self.functions[j].ddg(x)
        return _dgg

