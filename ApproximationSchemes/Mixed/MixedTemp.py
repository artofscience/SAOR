## Imports
import numpy as np
from ApproximationSchemes.NonMixed.Approx import Approximation
from ApproximationSchemes.NonMixed.Lin import Linear
from ApproximationSchemes.NonMixed.MMA import MMA


## InitiallyMixedFuncVar: Different approximation per response and variable set. Cannot switch during optimization
class MixedTemplate(Approximation):

    def __init__(self, n, m, xmin, xmax, **kwargs):
        Approximation.__init__(self, n, m, xmin, xmax, **kwargs)        # let parent class handle the common things
        self.reduced_array = kwargs.get('approx_array', None)           # array with approx names (resp_sets * var_sets)
        self.var_set = kwargs.get('var_set', None)                      # dictionary of different variable sets
        self.resp_set = kwargs.get('resp_set', None)                    # dictionary of different response sets
        self.num_of_var_sets = len(self.var_set.keys())                 # number of variable sets
        self.num_of_resp_sets = len(self.resp_set.keys())               # number of response sets
        self.name = 'MixedTemplate'

        # Create dictionary: self.obj_dict[j, i] = Linear(n, m, xmin, xmax)
        self.approx_obj = {}
        for j in range(0, self.num_of_resp_sets):
            for i in range(0, self.num_of_var_sets):

                if self.reduced_array[j, i] == 'Linear':

                    # Instantiate object linear at self.obj_dict[j, i]
                    self.approx_obj[j, i] = Linear(len(self.var_set[i]), len(self.resp_set[j]) - 1,
                                                   xmin[self.var_set[i]], xmax[self.var_set[i]])

                elif self.reduced_array[j, i] == 'MMA':

                    # Instantiate object mma at self.obj_dict[j, i]
                    self.approx_obj[j, i] = MMA(len(self.var_set[i]), len(self.resp_set[j]) - 1,
                                                xmin[self.var_set[i]], xmax[self.var_set[i]])

    ## Define intermediate vars:   y_ij := intermediate variable -i- of response function -j-
    def _set_y(self, x):
        y = np.zeros((self.n, self.m + 1))
        for j in range(0, self.num_of_resp_sets):
            for i in range(0, self.num_of_var_sets):
                y[np.ix_(self.var_set[i], self.resp_set[j])] = self.approx_obj[j, i]._set_y(x[self.var_set[i]])
        return y

    ## Define derivatives of intermediate vars:   dy_ij/dx_i
    def _set_dydx(self, x):
        dy = np.zeros((self.n, self.m + 1))
        for j in range(0, self.num_of_resp_sets):
            for i in range(0, self.num_of_var_sets):
                dy[np.ix_(self.var_set[i], self.resp_set[j])] = self.approx_obj[j, i]._set_dydx(x[self.var_set[i]])
        return dy

    ## Define 2nd-order derivatives of intermediate vars:   ddy_ij/ddx_i
    def _set_ddydx(self, x):
        ddy = np.zeros((self.n, self.m + 1))
        for j in range(0, self.num_of_resp_sets):
            for i in range(0, self.num_of_var_sets):
                ddy[np.ix_(self.var_set[i], self.resp_set[j])] = self.approx_obj[j, i]._set_ddydx(x[self.var_set[i]])
        return ddy

    ## Define chain rule term:   y = T_inv(x)  =>  x = T(y)  =>  dT/dy = dx/dy
    def _set_dTdy(self):
        dTdy = np.zeros((self.n, self.m + 1))
        for j in range(0, self.num_of_resp_sets):
            for i in range(0, self.num_of_var_sets):
                dTdy[np.ix_(self.var_set[i], self.resp_set[j])] = self.approx_obj[j, i]._set_dTdy()
        return dTdy

    ## Build current sub-problem for a mixed scheme: overrides Approximation.build_sub_prob(...)
    def build_sub_prob(self, x, g, dg):

        # Store current point
        self.x = x.copy()
        self.g = g.copy()
        self.dg = dg.copy()

        # Build sub-problems of the constituent approximations of a mixed scheme
        for j in range(0, self.num_of_resp_sets):
            for i in range(0, self.num_of_var_sets):
                self.approx_obj[j, i].build_sub_prob(x[self.var_set[i]], g[self.resp_set[j]],
                                                     dg[np.ix_(self.resp_set[j], self.var_set[i])])

        # Assembly for mixed schemes
        self._set_P()
        self._set_zo_term()
        self._set_bounds()

    ## Assemble P matrix for a mixed scheme: overrides Approximation._set_P()
    def _set_P(self):
        self.y_k = self._set_y(self.x)
        for j in range(0, self.num_of_resp_sets):
            for i in range(0, self.num_of_var_sets):
                self.P[np.ix_(self.resp_set[j], self.var_set[i])] = self.approx_obj[j, i].P       

    ## Use the most conservative bounds for a mixed scheme
    def _set_bounds(self):
        self.alpha = np.maximum.reduce([self.x - self.move_limit * self.dx, self.xmin])
        self.beta = np.minimum.reduce([self.x + self.move_limit * self.dx, self.xmax])
        for j in range(0, self.num_of_resp_sets):
            for i in range(0, self.num_of_var_sets):
                self.alpha[self.var_set[i]] = np.maximum(self.alpha[self.var_set[i]], self.approx_obj[j, i].alpha)
                self.beta[self.var_set[i]] = np.minimum(self.beta[self.var_set[i]], self.approx_obj[j, i].beta)

    ## Define some properties of the approximation scheme
    def _set_properties(self, **kwargs):
        self.properties.convex = kwargs.get('convex', None)

    ## Update old values
    def update_old_values(self, x, g, dg, itte, **kwargs):

        # Update parameters that are common for all approximations
        Approximation.update_old_values(self, x, g, dg, itte)

        # Update parameters for the constituent approximation members of a mixed scheme
        for j in range(0, self.num_of_resp_sets):
            for i in range(0, self.num_of_var_sets):
                self.approx_obj[j, i].iter = itte
                if self.iter > 1:
                    self.approx_obj[j, i].xold1 = self.xold1[self.var_set[i]]
                    self.approx_obj[j, i].xold2 = self.xold2[self.var_set[i]]
                    self.approx_obj[j, i].gold1 = self.gold1[self.resp_set[j]]
                    self.approx_obj[j, i].gold2 = self.gold2[self.resp_set[j]]
                    self.approx_obj[j, i].dgold1 = self.dgold1[np.ix_(self.resp_set[j], self.var_set[i])]
                    self.approx_obj[j, i].dgold2 = self.dgold2[np.ix_(self.resp_set[j], self.var_set[i])]
