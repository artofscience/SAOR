from abc import ABC, abstractmethod


class GBOpt(ABC):
    """
    Gradient-based optimizer.
    """

    def __init__(self, problem):
        self.n = problem.n
        self.m = problem.m

        self.x = problem.x

        self.g = problem.g
        self.dg = problem.dg
        self.ddg = problem.ddg

        self.alpha = problem.alpha
        self.beta = problem.beta


    @abstractmethod
    def update(self):
        ...