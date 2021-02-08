import numpy as np
import abc


class Approximation(abc.ABC):
    """Abstract approximation class.

    Provides base-class implementation for any approximation scheme.
    """

    # FIXME: which attributes are required for _all_ classes?
    # FIXME: is it required for `Approximation` to store the number of
    # design variables and constraints explicitly? Could we not only store
    # a reference to `problem` as `self.problem`?

    def __init__(self, problem, move_limit=0.1, **kwargs):
        # number of design variables and constraints
        self.n = problem.n
        self.m = problem.m

        # design variables, responses, sensitivities
        self.x = None
        self.g = None
        self.dg = None

        # previous iterates
        self.xold1 = None
        self.xold2 = None
        self.gold1 = None
        self.gold2 = None
        self.dgold1 = None
        self.dgold2 = None

        self.move_limit = move_limit
        self.iter = 0
        self.xmin = problem.xmin
        self.xmax = problem.xmax
        self.dx = self.xmax - self.xmin

        # constant part of Taylor expansion
        self.zo_term = np.zeros(self.m + 1)

        # dg/dx * dT/dy = dg/dx * dx/dy
        print(self.m, self.n)
        self.P = np.zeros((self.m + 1, self.n), dtype=float)

        # for the zero-order term
        self.y_k = np.zeros((self.n, self.m + 1), dtype=float)

        # e.g. non-convex, type, etc.
        self.properties = None

    def build_sub_problem(self, x, g, dg):
        self.x = x.copy()
        self.g = g.copy()
        self.dg = dg.copy()
        self._set_P()
        self._set_zo_term()
        self._set_bounds()

    # FIXME: function names

    @abc.abstractmethod
    def _set_y(self, x):
        ...

    @abc.abstractmethod
    def _set_dydx(self, x):
        ...

    @abc.abstractmethod
    def _set_ddydx(self, x):
        ...

    @abc.abstractmethod
    def _set_dTdy(self):
        ...

    # Set P matrix for non-mixed approximation schemes: P_ji =  dg/dx * dT/dy
    def _set_P(self):
        self.y_k = self._set_y(self.x)
        dTdy = self._set_dTdy()
        for j in range(0, self.m + 1):
            self.P[j, :] = self.dg[j, :] * dTdy[:, j]

    @abc.abstractmethod
    def _set_bounds(self):
        ...

    # Set the zero-order terms of responses -g_j- for the current iter:
    # zo_term := g_j(X^(k)) - P_ji^(k) * y_i(^(k))
    def _set_zo_term(self):
        self.zo_term = self.g.copy()
        for j in range(0, self.m + 1):
            self.zo_term[j] -= np.dot(self.P[j, :], self.y_k[:, j])
        self.b = - self.zo_term[1:]

    # Set the approximate response functions g_approx for the current iter
    # omitted zero-order term for solver
    def g_approx(self, x_curr):
        y = self._set_y(x_curr)
        g_approx_value = np.zeros(self.m + 1)
        for j in range(0, self.m + 1):
            g_approx_value[j] += np.dot(self.P[j, :], y[:, j])
        return g_approx_value

    # Set the approximate sensitivities dg_approx for the current iter
    def dg_approx(self, x_curr):
        dy = self._set_dydx(x_curr)
        dg_approx_value = np.zeros((self.m + 1, self.n))
        for j in range(0, self.m + 1):
            dg_approx_value[j, :] += self.P[j, :] * dy[:, j]
        return dg_approx_value

    # Set approximate 2nd-order sensitivities ddg_approx for the current iter
    def ddg_approx(self, x_curr):
        ddy = self._set_ddydx(x_curr)
        ddg_approx_value = np.zeros((self.m + 1, self.n))
        for j in range(0, self.m + 1):
            ddg_approx_value[j, :] += self.P[j, :] * ddy[:, j]
        return ddg_approx_value

    # FIXME: why does the approximation scheme needs to keep track of the
    # current iteration count?
    def update_old_values(self, x, g, dg, itte, **kwargs):
        self.iter = itte
        if self.iter > 1:
            self.xold2 = self.xold1.copy()
            self.gold2 = self.gold1.copy()
            self.dgold2 = self.dgold1.copy()
        self.xold1 = x.copy()
        self.gold1 = g.copy()
        self.dgold1 = dg.copy()
