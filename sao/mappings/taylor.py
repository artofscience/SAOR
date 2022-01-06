from .mapping import Mapping
from .exponential import Exponential
from sao.util.tools import parse_to_list
import numpy as np

class Taylor1(Mapping):
    """
    This class creates a 1st-order Taylor approximation, possibly using intervening variables.

    Without intervening variable:

    .. math::
        \\tilde{g}(x) = g(x_0) + \\left.\\frac{dg}{dx}\\right|_{x_0}(x - x_0)

    With intervening variable:

    .. math::
        \\tilde{g}(x) = g(x_0) + \\left.\\frac{dg}{dx}\\right|_{x_0}\\frac{dx}{dy}(y(x) - y(x_0))
    """

    def __init__(self, map=Exponential(1)):
        """Initialize the approximation, with optional intervening variable object."""
        self.map = map
        self.g0 = None
        self.y0 = None
        self.dgdy = None
        self.nresp, self.nvar = -1, -1

    def update(self, x, f, df, ddf=None):
        """Update the approximation with new information."""
        self.nresp, self.nvar = df.shape
        assert len(x) == self.nvar, "Mismatch in number of design variables."
        assert len(f) == self.nresp, "Mismatch in number of responses."
        self.map.update(x, f, df, ddf)
        self.g0 = f.copy()
        self.dgdy = df/self.map.dg(x)
        self.y0 = self.map.g(x)
        self.g0 = -self.dgdy*self.y0
        return self

    def g(self, x, out=None):
        """Evaluates the approximation at design point `x`."""
        if out is None:
            out = np.zeros(self.nresp)
        out[:] = self.g0
        out += self.dgdy * self.map.g(x)
        return out

    def dg(self, x, out=None):
        """Evaluates the approximation's gradient at design point `x`."""
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        else:
            out[:] = 0.
        out += self.dgdy * self.map.dg(x)
        return out

    def ddg(self, x, out=None):
        """Evaluates the approximation's second derivative at design point `x`."""
        if out is None:
            out = np.zeros((self.nresp, self.nvar))
        else:
            out[:] = 0.
        out += self.dgdy * self.map.ddg(x)
        return out

    def clip(self, x):
        """Clips any vector `x` within the feasible bounds of any intervening variables."""
        self.map.clip(x)
        return x
