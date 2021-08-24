from abc import ABC, abstractmethod


class Problem(ABC):
    """
    This is the abstract implementation of a problem.
    """
    def __init__(self):
        self.x_min, self.x_max = None, None
        self.x0 = None
        self.n, self.m = None, None

    @abstractmethod
    def g(self, x):
        ...

    @abstractmethod
    def dg(self, x):
        ...

    def ddg(self, x):
        ...
