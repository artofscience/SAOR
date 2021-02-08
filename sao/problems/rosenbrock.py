from scipy.optimize import rosen, rosen_der

from .response import Response
from .problem import Problem


class Rosen(Response):
    def f(self, x):
        return rosen(x)

    def df(self, x):
        return rosen_der(x)


class Rosenbrock(Problem):
    def __init__(self, n=1):
        super().__init__(Rosen(n))
