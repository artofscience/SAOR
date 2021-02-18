## Imports
import numpy as np
from ConvergenceCriteria.ConvCriterion import ConvergenceCriterion


## Contains all the possible convergence criteria one might use
class KKT(ConvergenceCriterion):

    ## Constructor of class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xmin = kwargs.get('xmin', None)
        self.xmax = kwargs.get('xmax', None)
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
        return KKT_res
