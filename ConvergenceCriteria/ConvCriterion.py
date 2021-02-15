
## Abstract class for convergence criteria. All criteria inherit from this class
class ConvergenceCriterion:
    
    ## Constructor of class
    def __init__(self, **kwargs):
        self.converged = False
        self.tolerance = 1e-4                       # The tolerance for the convergence criterion
        self.max_ite_opt = 100                      # Maximum number of optimization iterations
        self.name = None

    ## Method to calculate the value of the current convergence criterion (to be overriden)
    def get_response(self, **kwargs):
        return NotImplemented

    ## Max iteration criterion (it is used by all criteria)
    def max_iter(self, **kwargs):
        return kwargs.get('iter', None) >= self.max_ite_opt

    ## Default method to calculate if the convergence criterion is satisfied at the current iteration
    def assess_convergence(self, **kwargs):
        max_iter_flag = self.max_iter(**kwargs)
        if (self.get_response(**kwargs) < self.tolerance) or max_iter_flag:
            self.converged = True
            if max_iter_flag:
                print('Maximum number of {} iterations was reached'.format(kwargs.get('iter', '')))
            elif not kwargs.get('multi_criteria_flag', False):
                print('Criterion {} was satisfied within a tolerance of {}'.format(self.name, self.tolerance))
