## Imports
import numpy as np
import constants as ct
from ConvergenceCriteria.ConvCriterion import ConvergenceCriterion
from ConvergenceCriteria.KKT import KKT
from ConvergenceCriteria.ObjChange import ObjectivecChange
from ConvergenceCriteria.VarChange import VariableChange
from ConvergenceCriteria.MaxIteration import MaxIteration


## Contains all the possible convergence criteria one might use
class Alltogether(ConvergenceCriterion):

    ## Constructor of class
    def __init__(self, **kwargs):
        ConvergenceCriterion.__init__(self, **kwargs)
        self.name = 'Alltogether'
        self.kkt = KKT(**kwargs)
        self.obj_change = ObjectivecChange(**kwargs)
        self.var_change = VariableChange(**kwargs)
        self.max_iter = MaxIteration(**kwargs)
        self.criteria_list = [self.kkt, self.obj_change, self.var_change]       # Don't put MaxIteration in this list

    ## Function to calculate all the convergence criteria together are satisfied (apart from the maximum iteration)
    def assess_convergence(self, **kwargs):

        criteria_satisfaction = np.zeros(len(self.criteria_list), dtype=bool)
        for index, criterion in enumerate(self.criteria_list):
            criterion.assess_convergence(**kwargs)
            criteria_satisfaction[index] = criterion.converged
        self.max_iter.assess_convergence(**kwargs)
        if np.all(criteria_satisfaction) or self.max_iter.converged:
            self.converged = True
