## Imports
import numpy as np
import constants as ct
from ConvergenceCriteria.ConvCriterion import ConvergenceCriterion


## Contains all the possible convergence criteria one might use
class KKT(ConvergenceCriterion):

    ## Constructor of class
    def __init__(self, **kwargs):
        ConvergenceCriterion.__init__(self, **kwargs)
        self.name = 'KKT'

    ## Function to calculate |KKT|: Filter out bound constraints so that KKT goes to 0 when convergence is achieved
    def get_response(self, **kwargs):
        x_k = kwargs.get('x_k', None)
        dg = kwargs.get('dg', None)
        lam = kwargs.get('lam', None)
        KKT_red_idx = [index for index, value in enumerate(x_k)
                       if 1.001 * self.xmin[index] < value < 0.99 * self.xmax[index]]
        KKT_reduced = dg[0, KKT_red_idx] + np.dot(lam.T, dg[1:, KKT_red_idx])
        KKT_res = np.linalg.norm(KKT_reduced)
        # print(KKT_res)
        # print('\n', dg[0, :])
        # print('\n', dg[-1, :])
        # print('\n', lam)
        return KKT_res

    ## Function to calculate if KKT is satisfied at the current iteration (overrides default method)
    def assess_convergence(self, **kwargs):
        if self.get_response(**kwargs) < ct.TOLERANCE:
            self.converged = True
