from .convergence_criterion import ConvergenceCriterion
import numpy as np


class Feasibility(ConvergenceCriterion):
    """Asserts feasibility of constraints given in negative null form.

    Converges once all constraint values are smaller than a given tolerance.
    """
    def __init__(self, variable="constraints", target=1e-4):
        super().__init__(variable, target)

    def __call__(self, **kwargs):
        constraints = self.get_variable(**kwargs)
        self.converged = np.all(constraints < self.target)
