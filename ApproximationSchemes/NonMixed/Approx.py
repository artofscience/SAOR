## Imports
import numpy as np


## Abstract class, contains all common things
class Approximation:

    ## Constructor of class
    def __init__(self, n, m, xmin, xmax, **kwargs):
        self.n = n
        self.m = m                                                    # Number of constraints (g_j contains + objective)
        self.x = None                                                 # current design vars
        self.g = None                                                 # current responses
        self.dg = None                                                # current sensitivities
        self.ddg = None                                               # 2nd-order diagonal Hessian
        self.xold1 = None                                             # design vars of (k-1)
        self.xold2 = None                                             # design vars of (k-2)
        self.gold1 = None                                             # responses of (k-1)
        self.gold2 = None                                             # responses of (k-2)
        self.dgold1 = None                                            # sensitivities of (k-1)
        self.dgold2 = None                                            # sensitivities of (k-2)
        self.move_limit = 0.1
        self.iter = 0
        self.xmin = xmin
        self.xmax = xmax
        self.dx = xmax - xmin
        self.zo_term = np.zeros(self.m + 1)                           # constant part of Taylor expansion
        self.so = kwargs.get('second_order', False)                   # flag to add 2nd-order diagonal terms to Taylor
        self.P = np.zeros((self.m + 1, self.n), dtype=float)          # dg/dx * dT/dy = dg/dx * dx/dy
        if self.so:
            self.Q = np.zeros((self.m + 1, self.n), dtype=float)      # d^2g/dx^2 * d^2T/dy^2 = d^2g/dx^2 * d^2x/dy^2
        self.y_k = np.zeros((self.n, self.m + 1), dtype=float)        # for the zero-order term
        self.properties = None                                        # e.g. non-convex, type, etc.

    ## Build current sub-problem
    def build_sub_prob(self, x, g, dg, **kwargs):
        self.x = x.copy()
        self.g = g.copy()
        self.dg = dg.copy()
        self.y_k = self._set_y(self.x)
        self._set_P()
        if self.so:
            self.ddg = kwargs.get('ddg', None) 
            self._set_Q()
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
    def _set_ddTdy(self):
        return None

    ## Set P matrix for non-mixed approximation schemes: P_ji =  dg/dx * dT/dy
    def _set_P(self):
        dTdy = self._set_dTdy()
        for j in range(0, self.m + 1):
            self.P[j, :] = self.dg[j, :] * dTdy[:, j]

    ## Set Q matrix for non-mixed approximation schemes: Q_ji =  d^2g/dx^2 * d^2T/dy^2
    def _set_Q(self):
        ddTdy = self._set_ddTdy()
        for j in range(0, self.m + 1):
            self.Q[j, :] = self.ddg[j, :] * ddTdy[:, j]

    ## To be overriden by the approximation's method
    def _set_bounds(self):
        pass

    ## Set the zero-order terms of responses -g_j- for the current iter: zo_term := g_j(X^(k)) - P_ji^(k) * y_i(^(k))
    def _set_zo_term(self):
        self.zo_term = self.g.copy()
        for j in range(0, self.m + 1):
            self.zo_term[j] -= np.dot(self.P[j, :], self.y_k[:, j])
            if self.so:
                self.zo_term[j] += 0.5 * np.dot(self.Q[j, :], (self.y_k[:, j]) ** 2)
        self.b = - self.zo_term[1:]

    ## Set the approximate response functions g_approx for the current iter || omitted zero-order term for solver
    def g_approx(self, x_curr):
        y = self._set_y(x_curr)
        g_approx_value = np.zeros(self.m + 1)
        for j in range(0, self.m + 1):
            g_approx_value[j] += np.dot(self.P[j, :], y[:, j])
            if self.so:
                g_approx_value[j] += - np.dot(self.Q[j, :], (self.y_k[:, j] * y[:, j])) + \
                                     0.5 * np.dot(self.Q[j, :], (y[:, j] ** 2))
        return g_approx_value

    ## Set the approximate sensitivities dg_approx for the current iter
    def dg_approx(self, x_curr):
        dy = self._set_dydx(x_curr)
        dg_approx_value = np.zeros((self.m + 1, self.n))
        for j in range(0, self.m + 1):
            dg_approx_value[j, :] += self.P[j, :] * dy[:, j]
            if self.so:
                y = self._set_y(x_curr)
                dg_approx_value[j, :] += - self.Q[j, :] * self.y_k[:, j] * dy[:, j] + self.Q[j, :] * y[:, j] * dy[:, j]
        return dg_approx_value

    ## Set the approximate 2nd-order sensitivities ddg_approx for the current iter
    def ddg_approx(self, x_curr):
        ddy = self._set_ddydx(x_curr)
        ddg_approx_value = np.zeros((self.m + 1, self.n))
        for j in range(0, self.m + 1):
            ddg_approx_value[j, :] += self.P[j, :] * ddy[:, j]
            if self.so:
                y = self._set_y(x_curr)
                dy = self._set_dydx(x_curr)
                ddg_approx_value[j, :] += - self.Q[j, :] * self.y_k[:, j] * ddy[:, j] + \
                                          self.Q[j, :] * (dy[:, j] ** 2 + y[:, j] * ddy[:, j])
        return ddg_approx_value

    ## Update old values
    def update_old_values(self, x, g, dg, itte, **kwargs):
        self.iter = itte
        if self.iter > 1:
            self.xold2 = self.xold1.copy()
            self.gold2 = self.gold1.copy()
            self.dgold2 = self.dgold1.copy()
        self.xold1 = x.copy()
        self.gold1 = g.copy()
        self.dgold1 = dg.copy()
