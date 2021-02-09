## Imports
import numpy as np
import constants as ct
from ConvergenceCriteria.ConvCriterion import ConvergenceCriterion


## Maximum iteration convergence criterion
class MaxIteration(ConvergenceCriterion):           # TODO: Needs review on how to combine it with the other criteria

    ## Constructor of class
    def __init__(self, **kwargs):
        ConvergenceCriterion.__init__(self, **kwargs)
        self.name = 'Feasibility'

    ## Function to calculate if maximum iteration has been reached (overrides default method)
    def assess_convergence(self, **kwargs):
        g = kwargs.get('g', None)
        if np.all(g[1:] < ct.TOLERANCE):
            self.converged = True
