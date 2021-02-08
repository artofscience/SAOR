## Imports
import constants as ct
from ConvergenceCriteria.ConvCriterion import ConvergenceCriterion


## Maximum iteration convergence criterion
class MaxIteration(ConvergenceCriterion):

    ## Constructor of class
    def __init__(self, **kwargs):
        ConvergenceCriterion.__init__(self, **kwargs)
        self.name = 'MaxIteration'

    ## Function to calculate if maximum iteration has been reached (overrides default method)
    def assess_convergence(self, **kwargs):
        iteration = kwargs.get('iter', None)
        if iteration > ct.MAX_ITE_OPT:
            self.converged = True
