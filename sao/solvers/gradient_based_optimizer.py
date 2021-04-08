from abc import ABC, abstractmethod
import numpy as np

class GBOpt(ABC):
    """
    Gradient-based optimizer.
    """

    def __init__(self, problem, **kwargs):
        self.problem = problem

        self.n, self.m = self.problem.n, self.problem.m
        self.alpha, self.beta = self.problem.alpha, self.problem.beta

        self.x0 = kwargs.get('x0', 0.5*(self.alpha + self.beta))

    def g(self, x):
        return self.problem.g(x)

    def dg(self, x):
        return self.problem.dg(x)

    def ddg(self, x):
        return self.problem.ddg(x)

    def g_dg(self, x):
        return self.problem.g_dg(x)

    def g_dg_ddg(self, x):
        return self.problem.g_dg_ddg(x)

    @abstractmethod
    def update(self):
        ...
