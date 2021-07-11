from abc import ABC
from .gradient_based_optimizer import GBOpt
import warnings


class Primal(GBOpt, ABC):

    """
    Primal optimizer.
    """

    def __init__(self, problem):
        super().__init__(problem)
        if self.n > self.m:
            warnings.warn("You are using a primal optimizer, but the number of variables is large than the number of responses. "
                          "The update process will be slow. "
                          "Opt to use a dual or primal-dual optimizer.")
