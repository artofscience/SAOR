## Imports
from sao.problems.subproblem import Subproblem
import numpy as np


## Mixed scheme assembly class
class Mixed(Subproblem):

    def __init__(self, subprob_map, var_set, resp_set):
        self.num_of_var_sets = len(var_set.keys())                      # number of variable sets
        self.num_of_resp_sets = len(resp_set.keys())                    # number of response sets

        # Calculate n, m
        self.n = 0
        self.m = -1
        for l in range(0, self.num_of_var_sets):
            self.n += len(var_set[l])
        for p in range(0, self.num_of_resp_sets):
            self.m += len(resp_set[p])

        # Initialize arrays
        self.alpha = np.zeros(self.n, dtype=float)
        self.beta = np.ones(self.n, dtype=float)
        self.f = np.zeros(self.m + 1, dtype=float)
        self.df = np.zeros((self.m + 1, self.n), dtype=float)
        self.ddf = np.zeros((self.m + 1, self.n), dtype=float)

        # Store dictionaries
        self.subprob_map = subprob_map                                  # dictionary of subproblems
        self.var_set = var_set                                          # dictionary of variable sets
        self.resp_set = resp_set                                        # dictionary of response sets


    def build(self, x, f, df, ddf=None):
        self.n, self.m = len(x), len(f) - 1             # to fit Stijn's solvers
        self.f, self.df, self.ddf = f, df, ddf

        # Build constituent subproblems
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                if ddf is not None:
                    self.subprob_map[p, l].build(x[self.var_set[l]],
                                                 f[self.resp_set[p]],
                                                 df[np.ix_(self.resp_set[p], self.var_set[l])],
                                                 ddf[np.ix_(self.resp_set[p], self.var_set[l])])
                else:
                    self.subprob_map[p, l].build(x[self.var_set[l]],
                                                 f[self.resp_set[p]],
                                                 df[np.ix_(self.resp_set[p], self.var_set[l])])

        # Get mixed subproblem bounds
        self.get_bounds()

    def get_bounds(self):
        self.alpha[:] = 1e-1        # TODO: initialization of alpha, beta is problem dependent. You can do smth nicer!
        self.beta[:] = 1e2
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                self.alpha[self.var_set[l]] = np.maximum(self.alpha[self.var_set[l]],
                                                         self.subprob_map[p, l].alpha)
                self.beta[self.var_set[l]] = np.minimum(self.beta[self.var_set[l]],
                                                        self.subprob_map[p, l].beta)

    def g(self, x):
        g_value = - (self.num_of_var_sets - 1) * self.f      # cuz each time a var_set is added, so is the const. term
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                g_value[self.resp_set[p]] += self.subprob_map[p, l].approx.g(
                    self.subprob_map[p, l].inter.y(x[self.var_set[l]]).T)
        return g_value

    def dg(self, x):
        dg_value = np.zeros_like(self.df)
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                dg_value[np.ix_(self.resp_set[p], self.var_set[l])] = self.subprob_map[p, l].approx.dg(
                    self.subprob_map[p, l].inter.y(x[self.var_set[l]]).T,
                    self.subprob_map[p, l].inter.dydx(x[self.var_set[l]]))
        return dg_value

    def ddg(self, x):
        ddg_value = np.zeros_like(self.df)
        for p in range(0, self.num_of_resp_sets):
            for l in range(0, self.num_of_var_sets):
                ddg_value[np.ix_(self.resp_set[p], self.var_set[l])] = self.subprob_map[p, l].approx.ddg(
                    self.subprob_map[p, l].inter.y(x[self.var_set[l]]).T,
                    self.subprob_map[p, l].inter.dydx(x[self.var_set[l]]),
                    self.subprob_map[p, l].inter.ddyddx(x[self.var_set[l]]))
        return ddg_value
