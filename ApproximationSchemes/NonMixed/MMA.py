## Imports
import numpy as np
from ApproximationSchemes.NonMixed.Approx import Approximation


## MMA Approximation class
class MMA(Approximation):

    ## Constructor of class
    def __init__(self, n, m, xmin, xmax, **kwargs):
        Approximation.__init__(self, n, m, xmin, xmax)      # let parent class handle the common arguments

        # Initialization of MMA-specific parameters
        self.pijconst = 1e-3
        self.so = False                                         # False: No 2nd-order info || True: Use 2nd-order info
        self.dxmin = 1e-5
        self.albefa = 0.1                           # albefa is limiting the max change of vars wrt asymptotes. Was 0.1
        self.asyinit = 0.5                          # was 0.5
        self.asyincr = 1.2
        self.asydecr = 0.7
        self.epsimin = 1e-7
        self.low = self.xmin.copy()
        self.upp = self.xmax.copy()
        self.ittomax = 400
        self.factor = self.asyinit * np.ones(self.n)
        self.iterinitial = 1.5
        self.asybound = 10.0
        self.name = 'MMA'
        self.y_k = np.empty((self.n, self.m + 1))                # intermediate vars
        self.P = np.empty((self.m + 1, self.n))                  # P = max{dg/dx, 0} * dT/dy
        self.zo_term = np.empty(self.m + 1)                      # r_j^(k) in Svanberg's paper

    ## Define intermediate vars: each row corresponds to a branch of T_inv(x)
    def _set_y(self, x):
        y = np.empty((self.n, self.m + 1))
        for j in range(0, self.m + 1):
            y[self.dg[j, :] >= 0, j] = (1 / (self.upp - x))[self.dg[j, :] >= 0]
            y[self.dg[j, :] < 0, j] = 1 / (x - self.low)[self.dg[j, :] < 0]
        return y

    ## Define intermediate vars: each row corresponds to a branch of T_inv(x)
    def _set_dydx(self, x):
        dy = np.empty((self.n, self.m + 1))
        for j in range(0, self.m + 1):
            dy[self.dg[j, :] >= 0, j] = (1 / (self.upp - x) ** 2)[self.dg[j, :] >= 0]
            dy[self.dg[j, :] < 0, j] = (-1 / (x - self.low) ** 2)[self.dg[j, :] < 0]
        return dy

    ## Define intermediate vars: each row corresponds to a branch of T_inv(x)
    def _set_ddydx(self, x):
        ddy = np.empty((self.n, self.m + 1))
        for j in range(0, self.m + 1):
            ddy[self.dg[j, :] >= 0, j] = (2 / (self.upp - x) ** 3)[self.dg[j, :] >= 0]
            ddy[self.dg[j, :] < 0, j]  = (2 / (x - self.low) ** 3)[self.dg[j, :] < 0]
        return ddy

    ## Define chain rule term: y = T_inv(x) --> dT/dy = dx/dy
    def _set_dTdy(self):
        dTdy = np.empty((self.n, self.m + 1))
        for j in range(0, self.m + 1):
            dTdy[self.dg[j, :] >= 0, j] = (1 / self.y_k[:, j] ** 2)[self.dg[j, :] >= 0]
            dTdy[self.dg[j, :] < 0, j] = (-1 / self.y_k[:, j] ** 2)[self.dg[j, :] < 0]
        return dTdy

    ## Set asymptotes approx.low, approx.upp at current iteration
    def _set_asymptotes(self):

        # Initial values of asymptotes
        if self.iter < self.iterinitial:
            self.low = self.x - self.factor * self.dx  # if L_j = 0 & U_j = inf, then MMA = CONLIN
            self.upp = self.x + self.factor * self.dx

        # Go in here when k >= 3 and when you need to update asymptotes
        else:

            # depending on if the signs of (x_k-xold) and (xold-xold2) are opposite, indicating an oscillation in xi
            # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed

            # check for oscillations in variables (if zzz > 0: no oscillations, if zzz < 0: oscillations)
            zzz = (self.x - self.xold1) * (self.xold1 - self.xold2)

            # oscillating variables x_i are assigned a factor of asydecr and non-oscillating to asyincr
            self.factor[zzz > 0] = self.asyincr
            self.factor[zzz < 0] = self.asydecr

            # update lower and upper asymptotes
            self.low = self.x - self.factor * (self.xold1 - self.low)
            self.upp = self.x + self.factor * (self.upp - self.xold1)

            # check min and max bounds of asymptotes, as they cannot be too close or far from the variable (redundant?)
            lowmin = self.x - self.asybound * self.dx
            lowmax = self.x - 1 / (self.asybound ** 2) * self.dx
            uppmin = self.x + 1 / (self.asybound ** 2) * self.dx
            uppmax = self.x + self.asybound * self.dx

            # if given asymptotes cross boundaries put them to their max/min values (redundant?)
            self.low = np.maximum(self.low, lowmin)
            self.low = np.minimum(self.low, lowmax)
            self.upp = np.minimum(self.upp, uppmax)
            self.upp = np.maximum(self.upp, uppmin)

    ## Set approx.P, approx.Q matrices at current iteration
    def _set_P(self):
        self._set_asymptotes()
        dg_j_p = np.maximum(self.dg, 0)
        dg_j_m = np.minimum(self.dg, 0)
        self.y_k = self._set_y(self.x)
        dTdy = self._set_dTdy()
        for j in range(0, self.m + 1):
            self.P[j, :] = dg_j_p[j, :] * dTdy[:, j] + dg_j_m[j, :] * dTdy[:, j]

    ## Set the bounds alpha, beta for all variables -x- for the sub-problem generated by MMA
    def _set_bounds(self):

        # minimum variable bounds
        zzl1 = self.low + self.albefa * (self.x - self.low)  # limit change in x_i wrt asymptotes U_i, L_i
        zzl2 = self.x - self.move_limit * self.dx
        self.alpha = np.maximum.reduce([zzl1, zzl2, self.xmin])  # finds the max for each row of (zzl1, zzl2, xmin)

        # maximum variable bounds
        zzu1 = self.upp - self.albefa * (self.upp - self.x)     # limit change in x_i wrt asymptotes U_i, L_i
        zzu2 = self.x + self.move_limit * self.dx
        self.beta = np.minimum.reduce([zzu1, zzu2, self.xmax])  # finds the min for each row of (zzu1, zzu2, xmax)

    ## Define some properties of the approximation scheme
    def _set_properties(self, **kwargs):
        self.properties.convex = kwargs.get('convex', None)
        self.properties.separable = kwargs.get('separable', None)
