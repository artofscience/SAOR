from .approximation import Approximation
from sao.intervening_variables import Linear
from sao.util.tools import _parse_to_list
import numpy as np


# class GCMMA(Approximation):
#     def __init__(self, asyinit, asydecr, asyincr, asybound, albefa):
#         ...
#
#     def update(self, x, f, df, ddf=None):
#         calculate pij, qij
#
#     def g(self, x):
#         sum ()


class Taylor1(Approximation):
    def __init__(self, intervening=Linear()):
        self.interv = _parse_to_list(intervening)
        self.g0 = None
        self.y0 = None
        self.dgdy = None
        self.nresp, self.nvar = -1, -1

    def update(self, x, f, df, ddf=None):
        self.nresp, self.nvar = df.shape
        assert len(x) == self.nvar
        assert len(f) == self.nresp
        for intv in self.interv:
            intv.update(x, f, df, ddf)
        self.g0 = f*1
        self.dgdy = [df/intv.dydx(x) for intv in self.interv]
        self.y0 = [intv.y(x) for intv in self.interv]
        return self

    def g(self, x, out=None):
        delta_y = [intv.y(x) - y0i for intv, y0i in zip(self.interv, self.y0)]
        if out is None:
            out = np.zeros(self.nresp)
        out[:] = self.g0
        for dgdy_i, delta_yi in zip(self.dgdy, delta_y):
            out += np.sum(dgdy_i * delta_yi, axis=1)
        return out

    def dg(self, x, out=None):
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        for dgdy_i, intv in zip(self.dgdy, self.interv):
            out += dgdy_i * intv.dydx(x)
        return out

    def ddg(self, x, out=None):
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        for dgdy_i, intv in zip(self.dgdy, self.interv):
            out += dgdy_i * intv.ddyddx(x)
        return out

    def clip(self, x):
        [intv.clip(x) for intv in self.interv]
        return x

    # def g_and_dg(self, x):
    #     ...
    #
    # def g_and_dg_and_ddg(self, x):
    #     ...


class Taylor2(Taylor1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ddgddy = None

    def update(self, x, f, df, ddf=None):
        super().update(x, f, df, ddf)
        if ddf is None:
            raise RuntimeError("Second order taylor needs second order information")
        self.ddgddy = [ddf * intv.dxdy(x) ** 2 + df * intv.ddxddy(x) for intv in self.interv]

    def g(self, x, out=None):
        delta_y = [intv.y(x) - y0i for intv, y0i in zip(self.interv, self.y0)]
        if out is None:
            out = np.zeros(self.nresp)
        out[:] = self.g0
        for dgdy_i, delta_yi in zip(self.dgdy, delta_y):
            out += np.sum(dgdy_i * delta_yi, axis=1)
        for ddgddy_i, delta_yi in zip(self.ddgddy, delta_y):
            out += 0.5*np.sum(ddgddy_i * delta_yi**2, axis=1)
        return out

    def dg(self, x, out=None):
        delta_y = [intv.y(x) - y0i for intv, y0i in zip(self.interv, self.y0)]
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        super().dg(x, out=out)
        for i, intv in enumerate(self.interv):
            out += self.ddgddy[i] * delta_y[i] * intv.dydx(x)
        return out

        # def ddg(self, y, dy, ddy):
    #     return super().ddg(y, dy, ddy) + (self.ddfddy*(y-self.y).T)*ddy + self.ddfddy*dy**2

    def ddg(self, x, out=None):
        delta_y = [intv.y(x) - y0i for intv, y0i in zip(self.interv, self.y0)]
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        super().ddg(x, out=out)
        for i, intv in enumerate(self.interv):
            out += self.ddgddy[i] * (delta_y[i] * intv.ddyddx(x) + (intv.dydx(x)) ** 2)  # comment
        return out


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

    def update(self, x, y, f, df, dxdy, *args, **kwargs):
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
