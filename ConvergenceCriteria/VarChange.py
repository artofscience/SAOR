## Imports
import numpy as np
import constants as ct
from ConvergenceCriteria.ConvCriterion import ConvergenceCriterion


## Contains all the possible convergence criteria one might use
class VariableChange(ConvergenceCriterion):

    ## Constructor of class
    def __init__(self, **kwargs):
        ConvergenceCriterion.__init__(self, **kwargs)
        self.xmin = kwargs.get('xmin', None)
        self.xmax = kwargs.get('xmax', None)
        self.name = 'VariableChange'

    ## Function to calculate the Normalized Change in Design Variables
    def get_response(self, **kwargs):
        x_k = kwargs.get('x_k', None)
        xold1 = kwargs.get('xold1', None)
        if xold1 is not None:                                                                                           # TODO: Maybe its better with try-except
            VarChange_norm = np.linalg.norm((x_k - xold1) / (self.xmax - self.xmin))
        else:
            VarChange_norm = 1              # large value in order to continue loop
        return VarChange_norm
