## Imports
import numpy as np
from .ConvCriterion import ConvergenceCriterion
from .KKT import KKT
from .ObjChange import ObjectiveChange
from .VarChange import VariableChange
from .Feasibility import Feasibility


## Contains all the possible convergence criteria one might use
class Alltogether(ConvergenceCriterion):

    ## Constructor of class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kkt = KKT(**kwargs)
        self.obj_change = ObjectiveChange(**kwargs)
        self.var_change = VariableChange(**kwargs)
        self.feasibility = Feasibility(**kwargs)
        self.criteria_list = [self.kkt, self.obj_change, self.var_change, self.feasibility]

    ## Function to calculate all the convergence criteria together are satisfied (apart from the maximum iteration)
    def assess_convergence(self, **kwargs):
        criteria_satisfaction = np.zeros(len(self.criteria_list), dtype=bool)
        for index, criterion in enumerate(self.criteria_list):
            criterion.assess_convergence(multi_criteria_flag=True, **kwargs)
            criteria_satisfaction[index] = criterion.converged
        max_iter_flag = self.max_iter(**kwargs)
        if np.all(criteria_satisfaction) or max_iter_flag:
            self.converged = True
            if max_iter_flag:
                print('Maximum number of {} iterations was reached'.format(kwargs.get('iter', '')))
            else:
                print(f'Criterion {self.__class__.__name__} was satisfied within a tolerance of {self.tolerance}')