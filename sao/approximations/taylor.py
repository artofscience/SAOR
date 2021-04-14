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


# Multi-point Spherical Taylor expansion of Eq. 16: https://link.springer.com/article/10.1007/s00158-006-0070-6
class SphericalTaylor2(Taylor2):
    def __init__(self, **kwargs):
        super().__init__()
        self.x, self.xold1 = None, None
        self.f, self.fold1 = None, None
        self.dxdy, self.yold1 = None, None

    def update(self, x, y, f, df, dxdy, **kwargs):
        self.xold1 = self.x
        self.x = x
        self.fold1 = self.f
        self.f = f
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
        msg = (f'Expect sensitivity of size {self.m+1}x{self.n}: '
               f'Received {self.dfdy.shape}.')
        assert self.dfdy.shape == (self.m + 1, self.n), msg

        # If iter > 0, approximate curvature by forcing the curve to pass through xold1
        if self.xold1 is not None:
            self.ddfddy = self.get_curvature()
            msg = (f"Expected ddf size: {self.m+1}x{self.n}: "
                   f"Received: {self.ddfddy.shape}.")
            assert self.ddfddy.shape == (self.m + 1, self.n), msg
            if self.force_convex:
                self.ddfddy = self.enforce_convexity(self.ddfddy.copy())
        else:
            self.ddfddy = np.zeros_like(self.dfdy)
        return self

    # Calculate Eq. 16 for any intervening variable y(x)
    def get_curvature(self):
        if len(self.y.shape) == 1:
            dot_prod = np.dot(self.dfdy, (self.yold1 - self.y))
        else:
            dot_prod = np.einsum('ij,ji->i', self.dfdy, (self.yold1 - self.y))
        c_j = 2 * (self.fold1 - self.f - dot_prod) / sum((self.yold1 - self.y) ** 2)
        return np.broadcast_to(c_j, (self.y.shape[0], c_j.shape[0])).T


# Multi-point NonSpherical Taylor expansion of Eq. 23: https://link.springer.com/article/10.1007/s00158-006-0070-6
class NonSphericalTaylor2(Taylor2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x, self.xold1 = None, None
        self.f, self.fold1 = None, None
        self.dfold1, self.dxdy, self.yold1, self.idx = None, None, None, None

    def update(self, x, y, f, df, dxdy, **kwargs):
        self.xold1 = self.x
        self.x = x
        self.fold1 = self.f
        self.f = f
        self.dfold1 = self.df
        self.df = df
        self.y = y(x).T
        self.dfdy = df * dxdy(x)
        self.dxdy = dxdy
        if self.xold1 is not None:
            self.yold1 = y(self.xold1).T

        self.m = len(self.f) - 1
        self.n = len(self.x)

        # Calculate 2nd-order part
        ddf = kwargs.get('ddf', None)
        if ddf is not None:
            ddxddy = kwargs.get('ddxddy', None)
            self.ddfddy = ddf * dxdy(x) ** 2 + df * ddxddy(x)

        # Check dfdy for its dimensions
        msg = (f'Expect sensitivity of size {self.m+1}x{self.n}: '
               f'Received {self.dfdy.shape}.')
        assert self.dfdy.shape == (self.m + 1, self.n), msg

        # Calculate curvature
        if self.xold1 is not None:
            # For numerical stability, only do finite differences when |y_i - yold1_i| > 1e-4
            diff = abs(self.yold1 - self.y)
            if len(self.y.shape) == 1:
                self.idx = np.asarray(np.where(diff > 1e-4))[0, :]
            else:
                self.idx = np.argwhere(np.all(diff > 1e-4, axis=1))[:, 0]

            # Adjust the curvature @X^(k) by satisfying dg_j/dx_i @X^(k-1)
            self.ddfddy[:, self.idx] = self.get_curvature()

            # Check ddfddy for its dimensions
            msg = (f"Expected ddf size: {self.m+1}x{self.n}: "
                   f"Received: {self.ddfddy.shape}.")
            assert self.ddfddy.shape == (self.m + 1, self.n), msg

            # Enforce convexity of responses
            if self.force_convex:
                self.ddfddy = self.enforce_convexity(self.ddfddy.copy())
        else:
            self.ddfddy = np.zeros_like(self.dfdy)
        return self

    # Calculate Eq. 23 for any intervening variable y(x)
    def get_curvature(self):
        if len(self.y.shape) == 1:
            c_ji = (self.dfold1[:, self.idx] * self.dxdy(self.xold1)[self.idx] - self.dfdy[:, self.idx]) / \
                   (self.yold1[self.idx] - self.y[self.idx])
        else:
            c_ji = (self.dfold1[:, self.idx] * self.dxdy(self.xold1)[:, self.idx] - self.dfdy[:, self.idx]) / \
                   (self.yold1[self.idx, :] - self.y[self.idx, :]).T
        return c_ji


# GBMMA1 is a multi-point Taylor1 expansion of Eq. 15: https://link.springer.com/article/10.1007/s00158-002-0238-7
class GBMMA1(Taylor1):
    def __init__(self):
        super().__init__()
        self.x, self.xold1 = None, None
        self.df, self.dfold1 = None, None

    def update(self, x, y, f, df, dxdy, **kwargs):
        self.xold1 = self.x
        self.x = x
        self.f = f
        self.dfold1 = self.df
        self.df = df
        self.y = y(x).T

        # GBMMA1 satisfies gradient @X^(k-1)
        if self.xold1 is not None:
            self.dfdy = self.dfold1 * dxdy(self.xold1)
        else:
            self.dfdy = df * dxdy(x)

        # 2nd-order part
        ddf = kwargs.get('ddf', None)
        if ddf is not None:
            ddxddy = kwargs.get('ddxddy', None)
            self.ddfddy = ddf * dxdy(x) ** 2 + df * ddxddy(x)

        self.m = len(self.f) - 1
        self.n = len(self.y)

        msg = (f'Expect sensitivity of size {self.m + 1}x{self.n}: '
               f'Received {self.dfdy.shape}.')
        assert self.dfdy.shape == (self.m + 1, self.n), msg

        if self.ddfddy is not None:
            msg = (f"Expected ddf size: {self.m + 1}x{self.n}: "
                   f"Received: {self.ddfddy.shape}.")
            assert self.ddfddy.shape == (self.m + 1, self.n), msg

            if self.force_convex:
                self.ddfddy = self.enforce_convexity(self.ddfddy.copy())

        return self
