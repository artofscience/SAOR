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
    def __init__(self):
        super().__init__()
        self.x, self.xold1, self.f, self.fold1, self.dxdy, self.yold1 = None, None, None, None, None, None

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

        # 2nd-order part
        ddf = kwargs.get('ddf', None)
        if ddf is not None:
            ddxddy = kwargs.get('ddxddy', None)
            self.ddfddy = ddf * dxdy(x) ** 2 + df * ddxddy(x)

        self.m = len(self.f) - 1
        self.n = len(self.x)

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

    # TODO: dont use curvature when y-yold1 < 1e-3
    def get_curvature(self):
        dot = np.dot(self.dfdy, (self.yold1-self.y)) if len(self.y.shape) == 1 else np.einsum('ij,ji->i', self.dfdy, (self.yold1-self.y))
        c_j = 2 * (self.fold1 - self.f - dot) / sum((self.yold1 - self.y) ** 2)
        return np.broadcast_to(c_j, (self.y.shape[0], c_j.shape[0])).T


class NonSphericalTaylor2(Taylor2):
    def __init__(self):
        super().__init__()
        self.x, self.xold1, self.f, self.dfold1, self.dxdy = None, None, None, None, None

    def update(self, x, y, f, df, dxdy, **kwargs):
        self.xold1 = self.x
        self.x = x
        self.y = y
        self.f = f
        self.dfold1 = self.df
        self.df = df
        self.dfdy = df * dxdy(x)

        # 2nd-order part
        ddf = kwargs.get('ddf', None)
        if ddf is not None:
            ddxddy = kwargs.get('ddxddy', None)
            self.ddfddy = ddf * dxdy(x) ** 2 + df * ddxddy(x)

        self.m = len(self.f) - 1
        self.n = len(self.y)

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
                self.enforce_convexity(self.ddfddy)

        return self

    # TODO: Fix shape of ddfddy
    def get_curvature(self):
        return (self.dfold1 * self.dxdy(self.xold1) - self.dfdy) / (self.yold1 - self.y)
