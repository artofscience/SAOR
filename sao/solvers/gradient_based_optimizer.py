from abc import ABC, abstractmethod
import numpy as np

class GBOpt(ABC):
    """
    Gradient-based optimizer.
    """

    def __init__(self, problem, **kwargs):
        self.n = problem.n
        self.m = problem.m

        self.g = problem.g
        self.dg = problem.dg
        self.ddg = problem.ddg

        self.alpha = problem.alpha
        self.beta = problem.beta

        self.x0 = kwargs.get('x0', 0.5*(self.alpha + self.beta))

    @abstractmethod
    def update(self):
        ...
