from .problem import Problem
from scipy.optimize import rosen, rosen_der


class Rosenbrock(Problem):
    def __init__(self):
        super().__init__()

    def response(self, x):
        return rosen(x)

    def sensitivity(self, x):
        return rosen_der(x)
