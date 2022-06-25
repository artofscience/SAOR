import numpy as np
from .mapping import Mapping, Linear


class LinearApproximation(Mapping):
    """
    Linear Approximation (LA) f[x] of function g[x] at x0.

    f[x]    = g[x0] + g'[x0]*(x - x0)
            = (g[x0] - g'[x0]*x0) + g'[x0]*x

    f'[x]   = g'[x0]

    f''[x]  = 0
    """

    def __init__(self, mapping=Linear()):
        # Initialise the dependent mapping. If not provided, dependent mapping is Linear() which ends the chain.
        super().__init__(mapping)
        self.g0, self.dg0 = None, None

    def _update(self, x0, dg0, ddg0=0):
        self.g0 = -dg0 * x0
        self.dg0 = dg0
        return self._g(x0), self._dg(dg0), self._ddg(ddg0)  # Why not return self.g0, self.dg0, self.ddg0?

    def _g(self, x): return self.g0 + self.dg0 * x  # Excludes g[x0] (on purpose)

    def _dg(self, x): return self.dg0

    def _ddg(self, x): return np.zeros_like(x)


class DiagonalQuadraticApproximation(LinearApproximation):
    """
    Diagonal Quadratic Approximation (DQA) f[x] of function g[x] at x0.

    DQA builds on top of LA.
    Explanation of what meaning Diagonal Quadratic

    f[x]    = g[x0] + g'[x0]*(x - x0) + 1/2*g''[x0]*(x - x0)^2
            = (g[x0] - g'[x0]*x0 + 1/2*g''[x0]*x0^2) + (g'[x0] - g''[x0]*x0)*x + 1/2*g''[x0]*x^2
            = LA[x] + 1/2*g''[x0]*(x0^2 - 2*x0*x + x^2)
            = (LA[x] + 1/2*g''[x0]*x0^2) - g''[x0]*x0*x + 1/2*g''[x0]*x^2

    f'[x]   = (LA'[x] - g''[x0]*x0) + g''[x0]*x

    f''[x]  = LA''[x] + g''[x0], with LA''[x] = 0
    """

    def __init__(self, mapping=Linear()):
        super().__init__(mapping)
        self.ddg0 = None

    def _update(self, x0, dg0, ddg0=0):
        super()._update(x0, dg0)
        self.g0 += 0.5 * ddg0 * x0 ** 2
        self.dg0 -= ddg0 * x0
        self.ddg0 = ddg0

    def _g(self, x):
        """
        Function value of DQA function at x.

        f[x]    = g[x0] + g'[x0]*(x - x0) + 1/2*g''[x0]*(x - x0)^2
                = (g[x0] - g'[x0]*x0 + 1/2*g''[x0]*x0^2) + (g'[x0] - g''[x0]*x0)*x + 1/2*g''[x0]*x^2

        :param x: Incoming variable (or function) value
        :return: Function value at x
        """

        return self.g0 + self.dg0 * x + 0.5 * self.ddg0 * x ** 2

    def _dg(self, x):
        """
        First derivative of DQA function at x.

        f'[x] = (g'[x0] - g''[x0]*x0) + g''[x0]*x

        :param x: Incoming variable (or function) value
        :return: First derivative at x
        """

        return self.dg0 + self.ddg0 * x

    def _ddg(self, x):
        """
        Second derivative of DQA function at x.

        f''[x] = g''[x0]

        :param x: Incoming variable (or function) value
        :return: Second derivative at x
        """

        return self.ddg0
