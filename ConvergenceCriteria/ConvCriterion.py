## Imports
import constants as ct


## Abstract class for convergence criteria. All criteria inherit from this class
class ConvergenceCriterion:
    
    ## Constructor of class
    def __init__(self, **kwargs):
        self.converged = False

    ## Method to calculate the value of the current convergence criterion (to be overriden)
    def get_response(self, **kwargs):
        return None

    ## Max iteration criterion (it is used by all criteria)
    @staticmethod
    def max_iter(**kwargs):
        return kwargs.get('iter', None) >= ct.MAX_ITE_OPT

    ## Default method to calculate if the convergence criterion is satisfied at the current iteration
    def assess_convergence(self, **kwargs):
        max_iter_flag = self.max_iter(**kwargs)
        if (self.get_response(**kwargs) < ct.TOLERANCE) or max_iter_flag:
            self.converged = True
            if max_iter_flag:
                print('Maximum number of {} iterations was reached'.format(kwargs.get('iter', '')))
            elif not kwargs.get('multi_criteria_flag', False):
                print('Criterion {} was satisfied within a tolerance of {}'.format(self.name, ct.TOLERANCE))
