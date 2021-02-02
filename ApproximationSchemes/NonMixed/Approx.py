## Imports
import constants as ct
import numpy as np


## Abstract class, contains all common things
class Approximation:

    ## Constructor of class
    def __init__(self, n, m, xmin, xmax, **kwargs):
        self.n = n
        self.m = m                                              # Number of constraints (g_j contains + objective)
        self.x = None
        self.g = None
        self.dg = None
        self.xold1 = None                                       # design vars of (k-1)
        self.xold2 = None                                       # design vars of (k-2)
        self.gold1 = None                                       # responses of (k-1)
        self.gold2 = None                                       # responses of (k-2)
        self.dgold1 = None                                      # sensitivities of (k-1)
        self.dgold2 = None                                      # sensitivities of (k-2)
        self.move_limit = ct.MOVE_LIMIT                         # Alter from constants.py file
        self.iter = 0
        self.xmin = xmin
        self.xmax = xmax
        self.zo_term = np.zeros((self.m + 1))                   # zero-order term to be computed only once per iteration
        self.dx = xmax - xmin
        self.properties = None                                  # e.g. non-convex, type, etc.
        self.P = None
        self.y_k = None
        self.num_of_var_sets = 1                                # default number of variable sets
        self.num_of_resp_sets = 1                               # default number of response sets

    ## Build current sub-problem
    def build_sub_prob(self, x, g, dg):
        self.x = x.copy()
        self.g = g.copy()
        self.dg = dg.copy()
        self._set_P()
        self._set_zo_term()
        self._set_bounds()

    ## To be overriden by the approximation's method
    def _set_y(self, x):
        return None

    ## To be overriden by the approximation's method
    def _set_dydx(self, x):
        return None

    ## To be overriden by the approximation's method
    def _set_ddydx(self, x):
        return None

    ## To be overriden by the approximation's method
    def _set_dTdy(self):
        return None

    ## To be overriden by the approximation's method
    def _set_P(self):
        pass

    ## To be overriden by the approximation's method
    def _set_bounds(self):
        pass

    ## Set the zero-order term of all the responses -g- for the current iter
    def _set_zo_term(self):
        self.zo_term = self.g.copy()
        for j in range(0, self.m + 1):
            self.zo_term[j] -= np.dot(self.P[j, :], self.y_k[:, j])
        self.b = - self.zo_term[1:]

    ## Set the approximate response functions g_approx for the current iter || omitted zero-order term for solver
    def _g_approx(self, x_curr):
        y = self._set_y(x_curr)
        g_approx_value = np.zeros(self.m + 1)
        for j in range(0, self.m + 1):
            g_approx_value[j] += np.dot(self.P[j, :], y[:, j])
        return g_approx_value

    ## Set the approximate sensitivities dg_approx for the current iter || omitted zero-order term for solver
    def _dg_approx(self, x_curr):
        dy = self._set_dydx(x_curr)
        dg_approx_value = np.zeros((self.m + 1, self.n))
        for j in range(0, self.m + 1):
            dg_approx_value[j, :] += self.P[j, :] * dy[:, j]
        return dg_approx_value

    ## Set the approximate 2nd-order sensitivities ddg_approx for the current iter || omitted zero-order term for solver
    def _ddg_approx(self, x_curr):
        ddy = self._set_ddydx(x_curr)
        ddg_approx_value = np.zeros((self.m + 1, self.n))
        for j in range(0, self.m + 1):
            ddg_approx_value[j, :] += self.P[j, :] * ddy[:, j]
        return ddg_approx_value

    ## Update old values
    def update_old_values(self, x, g, dg, itte, **kwargs):
        self.iter = itte
        if self.iter > 0:
            self.xold2 = self.xold1.copy()
            self.gold2 = self.gold1.copy()
            self.dgold2 = self.dgold1.copy()
        self.xold1 = x.copy()
        self.gold1 = g.copy()
        self.dgold1 = dg.copy()
