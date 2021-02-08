## Imports
import numpy as np
import constants as ct
from ConvergenceCriteria.ConvCriterion import ConvergenceCriterion


## Contains all the possible convergence criteria one might use
class ObjectivecChange(ConvergenceCriterion):

    ## Constructor of class
    def __init__(self, **kwargs):
        ConvergenceCriterion.__init__(self, **kwargs)
        self.name = 'ObjectiveChange'

    ## Function to calculate the Normalized Change in Objective Function Values
    def get_response(self, **kwargs):
        g = kwargs.get('g', None)
        gold1 = kwargs.get('gold1', None)
        if gold1 is not None:
            if abs(gold1[0]) > 1e-5:
                ObjChange_abs = abs((g[0] - gold1[0]) / gold1[0])
            else:
                ObjChange_abs = abs((g[0] - gold1[0]))
        else:
            ObjChange_abs = 1       # large value in order to continue loop
        return ObjChange_abs
