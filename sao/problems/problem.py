from abc import ABC, abstractmethod


class Problem(ABC):
    """
    This is the abstract implementation of a problem.
    """
    def __init__(self):
        self.xmin = self.alpha = None
        self.xmax = self.beta = None
        self.x = None
        self.n, self.m = None, None

    @abstractmethod
    def g(self, x):
        ...

    @abstractmethod
    def dg(self, x):
        ...

    def ddg(self, x):
        ...

    def get_bounds(self):
        return self.xmin, self.xmax
