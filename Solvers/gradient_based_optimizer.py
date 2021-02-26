from abc import ABC, abstractmethod


class GBOpt(ABC):
    """
    Gradient-based optimizer.
    """

    def __init__(self, problem):
        self.x = problem.x
        self.zo = problem.zo
        self.g = problem.g
        self.dg = problem.dg
        self.ddg = problem.ddg
        self.alpha = problem.alpha
        self.beta = problem.beta
        self.n = problem.n
        self.m = problem.m

    @abstractmethod
    def update(self):
        ...
