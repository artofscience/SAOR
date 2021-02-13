from abc import ABC
from gradient_based_optimizer import GBOpt
import warnings


class Dual(GBOpt, ABC):

    """
    Dual optimizer.
    """

    def __init__(self, problem):
        super().__init__(problem)
        if self.m > self.n:
            warnings.warn("You are using a dual optimizer, but the number of responses is large than the number of variables. "
                          "The update process will be slow. "
                          "Opt to use a primal or primal-dual optimizer.")
