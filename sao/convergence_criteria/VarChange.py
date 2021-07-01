## Imports
import numpy as np
from .ConvCriterion import ConvergenceCriterion


## Contains all the possible convergence criteria one might use
class VariableChange(ConvergenceCriterion):

    ## Constructor of class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xmin = kwargs.get('xmin', None)
        self.xmax = kwargs.get('xmax', None)
        self.x_k, self.xold1 = None, None

    def get_response(self, **kwargs):
        """
        Function to calculate the Normalized Change in Design Variables.

        :param kwargs:
        :return:
        """

        self.xold1 = self.x_k
        self.x_k = kwargs.get('x_k', None)

        if self.xold1 is not None:                                          # TODO: Maybe its better with try-except
            VarChange_norm = np.linalg.norm((self.x_k - self.xold1) / (self.xmax - self.xmin))
        else:
            VarChange_norm = 1.0              # large value in order to continue loop
        self.value = VarChange_norm           # so you can access the criterion's value for logging/plotting purposes
        return VarChange_norm
