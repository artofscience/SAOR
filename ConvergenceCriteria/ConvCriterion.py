## Imports
import numpy as np
import constants as ct


## Abstract class for convergence criteria. All criteria inherit from this class
class ConvergenceCriterion:
    
    ## Constructor of class
    def __init__(self, **kwargs):
        self.converged = False

    ## Function to calculate the value of the current convergence criterion
    def get_response(self, **kwargs):
        return None

    ## Default function to calculate if the convergence criterion is satisfied at the current iteration
    def assess_convergence(self, **kwargs):
        if self.get_response(**kwargs) < ct.TOLERANCE:
            self.converged = True
