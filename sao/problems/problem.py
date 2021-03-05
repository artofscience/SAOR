from abc import ABC, abstractmethod


class Problem(ABC):
    """
    This is the abstract implementation of a problem.
    """
    def __init__(self):
        # TODO: All problems (and subproblems) must have alpha and beta, but only exact problems have xmin, xmax.
        #  Perhaps xmin, xmax can be removed completely, not sure.
        self.alpha, self.beta = None, None
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
        return self.alpha, self.beta
