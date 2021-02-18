## Imports
import numpy as np
from ConvergenceCriteria.ConvCriterion import ConvergenceCriterion


## Maximum iteration convergence criterion
class Feasibility(ConvergenceCriterion):

    ## Constructor of class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'Feasibility'

    ## Function to calculate if maximum iteration has been reached (overrides default method)
    def assess_convergence(self, **kwargs):
        g = kwargs.get('g', None)
        max_iter_flag = self.max_iter(**kwargs)
        if np.all(g[1:] < self.tolerance) or max_iter_flag:
            self.converged = True
            if max_iter_flag:
                print('Maximum number of {} iterations was reached'.format(kwargs.get('iter', '')))
            elif not kwargs.get('multi_criteria_flag', False):
                print('Criterion {} was satisfied within a tolerance of {}'.format(self.name, self.tolerance))
