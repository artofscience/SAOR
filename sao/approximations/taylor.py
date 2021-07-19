from .approximation import Approximation
from sao.intervening_variables import Linear
from sao.util.tools import parse_to_list
import numpy as np


class Taylor1(Approximation):
    """
    This class creates a 1st-order Taylor approximation, possibly using intervening variables.

    Without intervening variable:

    .. math::
        \\tilde{g}(x) = g(x_0) + \\left.\\frac{dg}{dx}\\right|_{x_0}(x - x_0)

    With intervening variable:

    .. math::
        \\tilde{g}(x) = g(x_0) + \\left.\\frac{dg}{dx}\\right|_{x_0}\\frac{dx}{dy}(y(x) - y(x_0))
    """

    def __init__(self, intervening=Linear()):
        """Initialize the approximation, with optional intervening variable object."""
        self.interv = parse_to_list(intervening)
        self.g0 = None
        self.y0 = None
        self.dgdy = None
        self.nresp, self.nvar = -1, -1

    def update(self, x, f, df, ddf=None):
        """Update the approximation with new information."""
        self.nresp, self.nvar = df.shape
        assert len(x) == self.nvar, "Mismatch in number of design variables."
        assert len(f) == self.nresp, "Mismatch in number of responses."
        for intv in self.interv:
            intv.update(x, f, df, ddf)
        self.g0 = f.copy()
        self.dgdy = [df/intv.dydx(x) for intv in self.interv]
        self.y0 = [intv.y(x) for intv in self.interv]
        return self

    def g(self, x, out=None):
        """Evaluates the approximation at design point `x`."""
        delta_y = [intv.y(x) - y for intv, y in zip(self.interv, self.y0)]
        if out is None:
            out = np.zeros(self.nresp)
        out[:] = self.g0
        for dgdy, dy in zip(self.dgdy, delta_y):
            out += np.sum(dgdy * dy, axis=1)
        return out

    def dg(self, x, out=None):
        """Evaluates the approximation's gradient at design point `x`."""
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        for dgdy, intv in zip(self.dgdy, self.interv):
            out += dgdy * intv.dydx(x)
        return out

    def ddg(self, x, out=None):
        """Evaluates the approximation's second derivative at design point `x`."""
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        for dgdy, intv in zip(self.dgdy, self.interv):
            out += dgdy * intv.ddyddx(x)
        return out

    def clip(self, x):
        """Clips any vector `x` within the feasible bounds of any intervening variables."""
        [intv.clip(x) for intv in self.interv]
        return x


class Taylor2(Taylor1):
    """
    Second order Taylor-approximation, with optional use of intervening variables.

    Without intervening variable:

    .. math::
        \\tilde{g}(x) = g(x_0) + \\left.\\frac{dg}{dx}\\right|_{x_0}(x - x_0) +
                        \\frac{1}{2}\\left.\\frac{d^2g}{dx^2}\\right|_{x_0}(x - x_0)^2

    With intervening variable:

    .. math::
        \\tilde{g}(x) = g(x_0) + \\left.\\frac{dg}{dx}\\right|_{x_0}\\left.\\frac{dx}{dy}\\right|_{y(x_0)}(y(x) - y(x_0))
                       + \\frac{1}{2}\\left(\\left.\\frac{d^2g}{dx^2}\\right|_{x_0}\\left(\\left.\\frac{dx}{dy}\\right|_{y(x_0)}\\right)^2
                       + \\left.\\frac{dg}{dx}\\right|_{x_0}\\left.\\frac{d^2x}{dy^2}\\right|_{y(x_0)}\\right)\\left(y(x) - y(x_0)\\right)^2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ddgddy = None

    def update(self, x, f, df, ddf=None):
        super().update(x, f, df, ddf)
        assert ddf is not None, "Second order taylor needs second order information"
        self.ddgddy = [ddf * intv.dxdy(x) ** 2 + df * intv.ddxddy(x) for intv in self.interv]

    def g(self, x, out=None):
        delta_y = [intv.y(x) - y for intv, y in zip(self.interv, self.y0)]
        if out is None:
            out = np.zeros(self.nresp)
        out[:] = self.g0  # Redo this part from Taylor1, to avoid recalculating delta_y
        for dgdy, dy in zip(self.dgdy, delta_y):
            out += np.sum(dgdy * dy, axis=1)
        for ddgddy, dy in zip(self.ddgddy, delta_y):
            out += 0.5*np.sum(ddgddy * dy**2, axis=1)
        return out

    def dg(self, x, out=None):
        delta_y = [intv.y(x) - y for intv, y in zip(self.interv, self.y0)]
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        super().dg(x, out=out)
        for i, intv in enumerate(self.interv):
            out += self.ddgddy[i] * delta_y[i] * intv.dydx(x)
        return out

    def ddg(self, x, out=None):
        delta_y = [intv.y(x) - y for intv, y in zip(self.interv, self.y0)]
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        super().ddg(x, out=out)
        for i, intv in enumerate(self.interv):
            out += self.ddgddy[i] * (delta_y[i] * intv.ddyddx(x) + (intv.dydx(x)) ** 2)
        return out


class SphericalTaylor2(Taylor2):
    """
    This is the Spherical 2nd-order Taylor expansion of Eq. 16 of the following paper:
    https://doi.org/10.1007/s00158-006-0070-6. It adjusts the 2nd-order diagonal information in the
    incomplete 2nd-order Taylor expansion to fit the previous point.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x, self.xold1 = None, None
        self.fold1 = None
        self.yold1 = None
        self.g0old1 = None

    def update(self, x, f, df, ddf=None):
        """
        This method updates the approximation instance for the SphericalTaylor2 expansion.

        :param x: Current design
        :param f: A vector of size [m+1] that holds the response values at the current design -x-
        :param df: A matrix of size [m+1, n] that holds the sensitivity values at the current design -x-
        :param kwargs: Optionally get the 2nd-order sensitivity array
        :return: self: For method cascading
        """
        self.g0old1 = self.g0
        self.xold1 = self.x
        self.x = x
        Taylor1.update(self, x, f, df, ddf)

        # If iter > 0, approximate curvature by using previous point information
        if self.xold1 is not None:
            self.yold1 = [intv.y(self.xold1) for intv in self.interv]
            self.set_curvature()
        else:
            self.ddgddy = [df*intv.ddxddy(x) for intv in self.interv]
        return self

    def set_curvature(self):
        """Approximate curvature by forcing the curve to pass through xold1, see Eq. 16."""
        if len(self.y0[0].shape) == 1:
            dot_prod = np.dot(self.dgdy[0], (self.yold1[0] - self.y0[0]))
        else:
            dot_prod = np.einsum('ij,ji->i', self.dgdy[0], (self.yold1[0] - self.y0[0]).T)
        c_j = 2 * (self.g0old1 - self.g0 - dot_prod) / sum((self.yold1[0] - self.y0[0]).T ** 2)
        self.ddgddy[0][:] = np.broadcast_to(c_j, (self.y0[0].T.shape[0], c_j.shape[0])).T
        return self


class NonSphericalTaylor2(SphericalTaylor2):
    """
    This is the NonSpherical 2nd-order Taylor expansion of Eq. 23 of the following paper:
    https://doi.org/10.1007/s00158-006-0070-6. It adjusts the 2nd-order diagonal information in the
    incomplete 2nd-order Taylor expansion to fit the previous point gradient.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df, self.dfold1 = None, None
        self.dxdyold1 = None
        self.idx = None
        self.epsi = 1e-2

    def update(self, x, f, df, ddf=None):
        """This method updates the approximation instance for the NonSphericalTaylor2 expansion."""
        self.g0old1 = self.g0
        self.xold1 = self.x
        self.x = x
        self.dfold1 = self.df
        self.df = df
        Taylor1.update(self, x, f, df, ddf)

        # If iter > 0, approximate curvature by using previous point information
        if self.xold1 is not None:
            self.yold1 = [intv.y(self.xold1) for intv in self.interv]
            self.dxdyold1 = [intv.dxdy(self.xold1) for intv in self.interv]
            self.set_curvature()
        else:
            self.ddgddy = [df * intv.ddxddy(x) for intv in self.interv]
        return self

    def set_curvature(self):
        """ Approximate curvature by forcing the curve to pass through xold1, see Eq. 23."""

        # For numerical stability, only do finite differences when |y_i - yold1_i| > self.epsi
        self.idx = abs(self.yold1[0] - self.y0[0]) > self.epsi
        if len(self.y0[0].shape) == 1:
            self.ddgddy[0][:, self.idx] = (self.dfold1[:, self.idx] * self.dxdyold1[0][self.idx] - self.dgdy[0][:, self.idx]) / \
                                          (self.yold1[0][self.idx] - self.y0[0][self.idx])
        else:
            self.ddgddy[0][self.idx] = (self.dfold1[self.idx] * self.dxdyold1[0][self.idx] - self.dgdy[0][self.idx]) / \
                                       (self.yold1[0][self.idx] - self.y0[0][self.idx])
        return self
