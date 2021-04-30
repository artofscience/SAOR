import numpy as np
from .ConvCriterion import ConvergenceCriterion


class Feasibility(ConvergenceCriterion):

    def assess_convergence(self, **kwargs):
        """
        Function to calculate if the variables are within the feasible domain (overrides default method).

        :param kwargs:
        :return:
        """

        f = kwargs.get('f', None)
        max_iter_flag = self.max_iter(**kwargs)
        if np.all(f[1:] < self.tolerance) or max_iter_flag:
            self.converged = True
            if max_iter_flag:
                print('Maximum number of {} iterations was reached'.format(kwargs.get('iter', '')))
            elif not kwargs.get('multi_criteria_flag', False):
                print(f'Criterion {self.__class__.__name__} was satisfied within a tolerance of {self.tolerance}')
