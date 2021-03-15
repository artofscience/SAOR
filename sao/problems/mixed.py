## Imports
from sao.problems.subproblem import Subproblem
import numpy as np


## InitiallyMixedFuncVar: Different approximation per response and variable set. Cannot switch during optimization
class Mixed:

    def __init__(self, n, m, subprob_dict, var_set, resp_set, **kwargs):
        self.alpha = np.zeros(n, dtype=float)
        self.beta = np.ones(n, dtype=float)
        self.f = np.zeros(m + 1, dtype=float)
        self.df = np.zeros((m + 1, n), dtype=float)
        self.ddf = np.zeros((m + 1, n), dtype=float)
        self.subprob_dict = subprob_dict                                # dictionary of subproblems
        self.var_set = var_set                                          # dictionary of different variable sets
        self.resp_set = resp_set                                        # dictionary of different response sets
        self.num_of_var_sets = len(self.var_set.keys())                 # number of variable sets
        self.num_of_resp_sets = len(self.resp_set.keys())               # number of response sets

    def build(self, x, f, df, ddf=None):
        self.n, self.m = len(x), len(f) - 1             # to fit Stijn's solvers
        self.f, self.df, self.ddf = f, df, ddf

        # Build constituent subproblems
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                self.subprob_dict[p, l].build(x[self.var_set[l]],
                                              f[self.resp_set[p]],
                                              df[[self.resp_set[p]], [self.var_set[l]]])

        # Get mixed subproblem bounds
        self.get_bounds()

    def get_bounds(self):
        self.alpha[:] = -1.
        self.beta[:] = 1.
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                self.alpha[self.var_set[l]] = np.maximum(self.alpha[self.var_set[l]],
                                                         self.subprob_dict[p, l].alpha)
                self.beta[self.var_set[l]] = np.minimum(self.beta[self.var_set[l]],
                                                        self.subprob_dict[p, l].beta)

    def g(self, x):
        g_value = - (self.num_of_var_sets - 1) * self.f      # cuz each time a var_set is added, so is the const. term
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                g_value[self.resp_set[p]] += self.subprob_dict[p, l].approx.g(
                    self.subprob_dict[p, l].inter.y(x[self.var_set[l]]).T)
        return g_value

    def dg(self, x):
        dg_value = np.zeros_like(self.df)
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                dg_value[[self.resp_set[p]], [self.var_set[l]]] = self.subprob_dict[p, l].approx.dg(
                    self.subprob_dict[p, l].inter.y(x[self.var_set[l]]).T,
                    self.subprob_dict[p, l].inter.dydx(x[self.var_set[l]]))
        return dg_value

    def ddg(self, x):
        ddg_value = np.zeros_like(self.df)
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                ddg_value[[self.resp_set[p]], [self.var_set[l]]] = self.subprob_dict[p, l].approx.ddg(
                    self.subprob_dict[p, l].inter.y(x[self.var_set[l]]).T,
                    self.subprob_dict[p, l].inter.dydx(x[self.var_set[l]]),
                    self.subprob_dict[p, l].inter.ddyddx(x[self.var_set[l]]))
        return ddg_value
