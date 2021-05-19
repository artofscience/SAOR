## IMPORTS
from abc import ABC, abstractmethod


## Abstract class for convergence criteria. All criteria inherit from this class
class ConvergenceCriterion(ABC):
    """ Abstract implementation of a convergence criterion.
        The class should provide the the `get_response`, `max_iter` and `assess_convergence` methods.
    """
    
    def __init__(self, **kwargs):
        self.converged = False
        self.tolerance = 1e-4                       # The tolerance for the convergence criterion
        self.max_ite_opt = 500                      # Maximum number of optimization iterations

    def get_response(self, **kwargs):
        """
        Method to calculate the value of the current convergence criterion (to be overriden).

        :param kwargs:
        :return:
        """

        return NotImplemented

    def max_iter(self, **kwargs):
        """
        Max iteration criterion (it is used by all criteria).

        :param kwargs:
        :return:
        """

        current_iter = kwargs.get('iter')
        if current_iter is not None:
            return current_iter >= self.max_ite_opt
        else:
            raise Exception('You must pass the iteration number to criterion.assess_convergence')

    def assess_convergence(self, **kwargs):
        """
        Default method to calculate if the convergence criterion is satisfied at the current iteration.

        :param kwargs:
        :return:
        """

        max_iter_flag = self.max_iter(**kwargs)
        if (self.get_response(**kwargs) < self.tolerance) or max_iter_flag:
            self.converged = True
            if max_iter_flag:
                print(r'Maximum number of {} iterations was reached'.format(kwargs.get('iter', '')))
            elif not kwargs.get('multi_criteria_flag', False):
                print(f'Criterion {self.__class__.__name__} was satisfied within a tolerance of {self.tolerance}')
