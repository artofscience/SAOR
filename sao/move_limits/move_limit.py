import math
from abc import ABC, abstractmethod


class MoveLimitStrategy(ABC):
    """
    This is an abstract implementation of a move limit strategy.
    """
    def __init__(self, xmin=-math.inf, xmax=math.inf, move_limit=0.1, **kwargs):
        """
        Description.
        :param xmin:
        :param xmax:
        :param move_limit:
        :param kwargs:
        """
        self.xmin = xmin
        self.xmax = xmax
        self.move_limit = move_limit

    @abstractmethod
    def update(self, x, **kwargs):
        ...
