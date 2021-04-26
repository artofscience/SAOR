from abc import ABC
from .gradient_based_optimizer import GBOpt


class PrimalDual(GBOpt, ABC):

    """
    Primal-dual optimizer.
    """

    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)
