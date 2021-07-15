from .convergence_criterion import ConvergenceCriterion


class MaximumIteration(ConvergenceCriterion):
    def __init__(self, variable="itte", target=500):
        super().__init__(variable, target)

    def __call__(self, **kwargs):
        iteration = self.get_variable(**kwargs)
        self.converged = iteration > self.target
