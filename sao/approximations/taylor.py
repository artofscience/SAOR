from .approximation import Approximation
import numpy as np


class Taylor1(Approximation):
    def g(self, y):
        return self.f + (np.dot(self.dfdy, (y-self.y)) if len(y.shape) == 1 else np.einsum('ij,ji->i', self.dfdy, (y-self.y)))

    def dg(self, y, dy):
        return self.dfdy * dy

    def ddg(self, y, dy, ddy):
        return self.dfdy * ddy


class Taylor2(Taylor1):
    def g(self, y):
        return super().g(y) + 0.5 * (np.dot(self.ddfddy, (y-self.y)**2) if len(y.shape) == 1 else np.einsum('ij,ji->i', self.ddfddy, (y-self.y)**2))

    def dg(self, y, dy):
        return super().dg(y, dy) + (self.ddfddy * (y-self.y).T) * dy

    def ddg(self, y, dy, ddy):
        return super().ddg(y, dy, ddy) + (self.ddfddy*(y-self.y).T)*ddy + self.ddfddy*dy**2


class SphericalTaylor2(Taylor2):
    """
    This is the Spherical 2nd-order Taylor expansion of Eq. 16 of the following paper:
    https://link.springer.com/article/10.1007/s00158-006-0070-6 .
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x, self.xold1 = None, None
        self.fold1 = None
        self.dfold1 = None
        self.dxdy, self.yold1 = None, None

    def update(self, x, y, f, df, dxdy, **kwargs):
        """
        This method updates the approximation instance for multi-point (approximate) 2nd-order Taylor expansions.

        :param x: Current design
        :param y: A method that returns the intervening variables at the current design, i.e. y(x)
        :param f: A vector of size [m+1] that holds the response values at the current design -x-
        :param df: A matrix of size [m+1, n] that holds the sensitivity values at the current design -x-
        :param dxdy: A method that returns the derivative of the inverse intervening variable function, i.e. dx/dy(y(x))
        :param kwargs: Optionally get the 2nd-order sensitivity array
        :return: self: For method cascading
        """

        self.xold1 = self.x
        self.x = x
        self.fold1 = self.f
        self.f = f
        self.dfold1 = self.df       # for NonSphericalTaylor2
        self.df = df                # for NonSphericalTaylor2
        self.y = y(x).T
        self.dfdy = df * dxdy(x)
        self.dxdy = dxdy
        if self.xold1 is not None:
            self.yold1 = y(self.xold1).T

        self.m = len(self.f) - 1
        self.n = len(self.x)

        # Calculate 2nd-order part and check dfdy for its dimensions
        ddf = kwargs.get('ddf', None)
        if ddf is not None:
            ddxddy = kwargs.get('ddxddy', None)
            self.ddfddy = ddf * dxdy(x) ** 2 + df * ddxddy(x)

        # If iter > 0, approximate curvature by using previous point information
        if self.xold1 is None:
            self.ddfddy = np.zeros_like(self.dfdy)
        else:
            self.set_curvature()

        # Check size of dfdy and (optionally) ddfddy
        self.check_sensitivity_sizes()

        if self.force_convex:
            self.enforce_convexity()

        return self

    def set_curvature(self):
        """
        Approximate curvature by forcing the curve to pass through xold1, see Eq. 16.

        :return: self: For method cascading
        """
        if len(self.y.shape) == 1:
            dot_prod = np.dot(self.dfdy, (self.yold1 - self.y))
        else:
            dot_prod = np.einsum('ij,ji->i', self.dfdy, (self.yold1 - self.y))
        c_j = 2 * (self.fold1 - self.f - dot_prod) / sum((self.yold1 - self.y) ** 2)
        self.ddfddy[:] = np.broadcast_to(c_j, (self.y.shape[0], c_j.shape[0])).T

        return self



class NonSphericalTaylor2(SphericalTaylor2):
    """
    This is the NonSpherical 2nd-order Taylor expansion of Eq. 23 of the following paper:
    https://link.springer.com/article/10.1007/s00158-006-0070-6 .
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.idx = None
        self.epsi = 1e-2

    def set_curvature(self):
        """
        Approximate curvature by forcing the curve to pass through xold1, see Eq. 23.

        :return: self: For method cascading
        """

        # For numerical stability, only do finite differences when |y_i - yold1_i| > self.epsi
        diff = abs(self.yold1 - self.y)
        if len(self.y.shape) == 1:
            self.idx = np.asarray(np.where(diff > self.epsi))[0, :]
        else:
            self.idx = np.argwhere(np.all(diff > self.epsi, axis=1))[:, 0]
        if len(self.y.shape) == 1:
            self.ddfddy[:, self.idx] = (self.dfold1[:, self.idx] * self.dxdy(self.xold1)[self.idx] - self.dfdy[:, self.idx]) / \
                                       (self.yold1[self.idx] - self.y[self.idx])
        else:
            self.ddfddy[:, self.idx] = (self.dfold1[:, self.idx] * self.dxdy(self.xold1)[:, self.idx] - self.dfdy[:, self.idx]) / \
                                       (self.yold1[self.idx, :] - self.y[self.idx, :]).T

        return self
