import math
from abc import ABC, abstractmethod


class MoveLimitStrategy(ABC):
    def __init__(self, xmin=-math.inf, xmax=math.inf, move_limit=0.1, **kwargs):
        self.xmin = xmin
        self.xmax = xmax
        self.move_limit = move_limit

    @abstractmethod
    def update_move_limit(self, x, **kwargs):
        ...
