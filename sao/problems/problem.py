from abc import ABC, abstractmethod


class Problem(ABC):
    def __init__(self, n, m, xmin, xmax):
        self.n = n
        self.m = m
        self.xmin = xmin
        self.xmax = xmax

    @abstractmethod
    def response(self, x):
        ...

    @abstractmethod
    def sensitivity(self, x):
        ...
