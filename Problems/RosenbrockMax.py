import numpy as np
from Problems.AbstractResponse import Response
from Problems.AbstractProblem import Problem


class RosenObjective(Response):
    def f(self, x_k):
        func_value = (2.5 - x_k[0]) ** 2 + 100. * ((x_k[1] - 0.5) - (x_k[0] - 1.5) ** 2) ** 2
        return func_value

    def df(self, x_k):
        sens_value = np.empty(len(x_k))
        sens_value[0] = 2.0 * x_k[0] + 506.25 * (2.66666666666667 - 1.77777777777778 * x_k[0]) * \
                     (0.444444444444444 * x_k[1] - (0.666666666666667 * x_k[0] - 1) ** 2 - 0.222222222222222) - 5.
        sens_value[1] = 200.0 * x_k[1] - 450.0 * (0.666666666666667 * x_k[0] - 1) ** 2 - 100.0
        return sens_value


class RosenConstraint1(Response):
    def f(self, x_k):
        func_value = (x_k[0] - 2.5) ** 3 - x_k[1] + 1.5
        return func_value

    def df(self, x_k):
        sens_value = np.empty(len(x_k))
        sens_value[0] = 3 * (x_k[0] - 2.5) ** 2
        sens_value[1] = -1
        return sens_value


class RosenConstraint2(Response):
    def f(self, x_k):
        func_value = x_k[0] + x_k[1] - 4.
        return func_value

    def df(self, x_k):
        sens_value = np.empty(len(x_k))
        sens_value[0] = 1
        sens_value[1] = 1
        return sens_value


class Rosenbrock(Problem):
    def __init__(self, **kwargs):
        self.n = 2
        xmin = np.array([0., 0.])
        xmax = np.array([3., 3.])
        x_0 = np.array([0.5, 0.3])
        Problem.__init__(self, [RosenObjective(), RosenConstraint1(), RosenConstraint2()], xmin, xmax, x_init=x_0)
