from abc import ABC, abstractmethod


class Approximation(ABC):
    def __init__(self):
        self.x = None
        self.xmin, self.xmax = None, None  # Bounds of the subproblem
        self.f, self.df, self.ddf = None, None, None

        self.m = -1  # Number of constraints
        self.n = -1  # Number of variables

    def update_approximation(self, x, xmin, xmax, f, df, ddf=None):
        self.x = x
        self.xmin, self.xmax = xmin, xmax
        self.f, self.df, self.ddf = f, df, ddf

        self.m = len(self.f) - 1
        self.n = len(self.x)
        assert self.df.shape == (self.m + 1, self.n), "Sensitivities should be of size (m+1 x n)"
        if self.ddf is not None:
            assert self.ddf.shape == (self.m + 1, self.n), "Second order derivatives should be of size (m+1 x n) (diagonal entries)"

        self.build_approximation()
        return self

    def build_approximation(self):
        pass

    @abstractmethod
    def g(self, x):
        ...

    @abstractmethod
    def dg(self, x):
        ...

    @abstractmethod
    def ddg(self, x):
        ...


