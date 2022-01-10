from abc import ABC, abstractmethod
import numpy as np
from sao.move_limits.bounds import Bounds
from sao.util.tools import parse_to_list


class Mapping(ABC):

    @property
    def name(self):
        return self.__class__.name

    @abstractmethod
    def g(self, x):
        """Approximate response function."""
        ...

    @abstractmethod
    def dg(self, x):
        """Approximate sensitivity array."""
        ...

    @abstractmethod
    def ddg(self, x):
        """Approximate 2nd-order sensitivity array."""
        ...

    def g_and_dg(self, x):
        return self.g(x), self.dg(x)

    def g_and_dg_and_ddg(self, x):
        return self.g(x), self.dg(x), self.ddg(x)


class Problem(Mapping, ABC):
    """
    This is the abstract implementation of a problem.
    """

    def __init__(self):
        self.x_min, self.x_max = None, None
        self.x0 = None
        self.n, self.m = None, None


class Approximation(Mapping):
    '''
    Approximation is a function mapping f: R^n -> R
    '''

    @abstractmethod
    def update(self, x, f, df, ddf=None):
        """
        This method updates the approximation instance.

        :param x: Current design
        :param f: A vector of size [m+1] that holds the response values at the current design -x-
        :param df: A matrix of size [m+1, n] that holds the sensitivity values at the current design -x-
        :param ddf: Optionally get the 2nd-order sensitivity array
        :return: self: For method cascading
        """
        ...

    def clip(self, x):
        return x


class Intervening(Approximation, ABC):
    """Abstract base class for the intervening variable mapping.

    This class provides a change of variables from y = f(x), transforming the
    variables x to y using a given transformation function f. Any child class
    should provide the functionality to compute the mapping y = f(x), as well
    as the first and second derivatives. Additionally, the inverse mapping
    should be provided, reversing the transformation.

    For details on the formulation, in specific regarding the first and
    second derivatives of the mapping and their inverses, see the reference
    material at: `reference_files/TaylorExpansion.pdf`.
    """

    def dx(self, x):
        """Evaluates the first derivative of the inverse mapping at x.

        For details refer to the reference material provided at:
        `reference_files/TaylorExpansion.pdf`
        `https://www.physicsforums.com/threads/is-the-derivative-equal-to-one-over-the-derivative-of-the-inverse.63886/`
        """
        return 1 / self.dg(x)

    def ddxddy(self, x):
        """Evaluates the second derivative of the inverse mapping at x.
        For details refer to the reference material provided at:
        `http://www.math-principles.com/2014/03/second-derivative-problems-reciprocal.html`
        """
        return -self.ddg(x) / self.dg(x) ** 3


# class Subproblem(Problem):
#     def __init__(self, approximation=[], limits=Bounds(xmin=0, xmax=1)):
#         super().__init__()
#         self.approx = approximation
#         self.lims = None
#         self.set_limits(limits)
#
#     def set_limits(self, *limits):
#         self.lims = parse_to_list(*limits)
#
#     def add_limits(self, *limits):
#         self.lims.extend(parse_to_list(*limits))
#
#     def build(self, x, f, df, ddf=None):
#         self.n, self.m = len(x), len(f) - 1
#
#         # Update the approximation
#         self.approx.update(x, f, df, ddf)
#
#         # Update the local problem bounds
#         self.x_min = np.full_like(x, -np.inf)
#         self.x_max = np.full_like(x, +np.inf)
#
#         # Enforce restriction on the possible step size within the subproblem.
#         # The step is restricted by the chosen move limit strategy as well as
#         # the feasible range of the intervening variables. First the move
#         # limits are applied to constraint the step size.
#         for ml in self.lims:
#             ml.update(x, f, df, ddf)
#             ml.clip(self.x_min)
#             ml.clip(self.x_max)
#
#         # Additional constraint on the step size by the feasible range of the
#         # intervening variables. This prevents the subsolver to make an update
#         # that causes the intervening variable to reach unreachable values,
#         # e.g. cross the lower/upper bounds in the MMA asymptotes.
#         self.approx.clip(self.x_min)
#         self.approx.clip(self.x_max)
#
#         assert np.isfinite(self.x_min).all() and np.isfinite(self.x_max).all(), \
#             "The bounds must be finite. Use at least one move-limit or bound."


class Exponential(Intervening):
    """A generic exponential intervening variable y = x^p.

    The general case for an exponential intervening varaibles that can take
    on various forms depending on the chosen power. Note: the implementation
    does not support ``p = 0`` to avoid a zero devision in the derivatives.
    """

    def __init__(self, mapping=None, p=1, xlim=1e-10):
        """
        Initialise the exponential intervening variable with a power.
        :param p: The power
        :param xlim: Minimum x, in case of negative p, to prevent division by 0
        """
        assert p != 0, f"Invalid power x^{p}, will result in zero division."
        self.p = p
        self.xlim = xlim
        self.map = mapping

    def update(self, x, f, df, ddf=None):
        if self.map is not None:
            self.map.update(x, f, df, ddf)

    def g(self, x):
        if self.map is not None:
            return self.map.g(x) ** self.p
        else:
            return x ** self.p

    def dg(self, x):
        if self.map is not None:
            return self.p * x ** (self.p - 1) * self.map.dg(x)
        else:
            return self.p * x ** (self.p - 1)

    def ddg(self, x):
        if self.map is not None:
            return self.p * (self.p - 1) * x ** (self.p - 2) * self.map.ddg(x)
        else:
            return self.p * (self.p - 1) * x ** (self.p - 2)

    def clip(self, x):
        if self.map is not None:
            x = self.map.clip(x)
        if self.p < 0:
            return np.maximum(x, self.xlim, out=x)
        return x


class Taylor1(Approximation):
    def __init__(self, mapping=Exponential(p=1)):
        """Initialize the approximation, with optional intervening variable object."""
        self.map = mapping
        self.g0 = None
        self.dgdy0 = None
        self.nresp, self.nvar = -1, -1

    def update(self, x, f, df, ddf=None):
        """Update the approximation with new information."""
        self.nresp, self.nvar = df.shape
        assert len(x) == self.nvar, "Mismatch in number of design variables."
        assert len(f) == self.nresp, "Mismatch in number of responses."
        self.map.update(x, f, df, ddf)
        self.dgdy0 = df / self.map.dg(x)
        self.g0 = (f / self.nvar)[:, np.newaxis] - self.dgdy0 * self.map.g(x)

    def g(self, x):
        """Evaluates the approximation at design point `x`."""
        return self.g0 + self.dgdy0 * self.map.g(x)

    def dg(self, x):
        """Evaluates the approximation's gradient at design point `x`."""
        return self.dgdy0 * self.map.dg(x)

    def ddg(self, x):
        """Evaluates the approximation's second derivative at design point `x`."""
        return self.dgdy0 * self.map.ddg(x)

    def clip(self, x):
        """Clips any vector `x` within the feasible bounds of any intervening variables."""
        return self.map.clip(x)
