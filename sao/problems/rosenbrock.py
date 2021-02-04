import numpy as np

from .problem import Problem
from scipy.optimize import rosen, rosen_der


class Rosenbrock(Problem):
    def __init__(self):
        super().__init__(1, 1, np.array(0), np.array(3))

    def response(self, x):
        return rosen(x)

    def sensitivity(self, x):
        return rosen_der(x)
