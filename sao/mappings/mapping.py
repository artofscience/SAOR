from abc import ABC, abstractmethod
import numpy as np
from sao.move_limits.bounds import Bounds
from sao.mappings.taylor import Taylor1
from sao.util.tools import parse_to_list


class Mapping(ABC):

    def __init__(self):
        self.name = "Default name"

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def g(self, x, out=None):
        """Approximate response function."""
        ...

    @abstractmethod
    def dg(self, x, out=None):
        """Approximate sensitivity array."""
        ...

    @abstractmethod
    def ddg(self, x, out=None):
        """Approximate 2nd-order sensitivity array."""
        ...

    def dx(self, x):
        """Evaluates the first derivative of the inverse mapping at x.

        For details refer to the reference material provided at:
        `ReferenceFiles/TaylorExpansion.pdf`
        `https://www.physicsforums.com/threads/is-the-derivative-equal-to-one-over-the-derivative-of-the-inverse.63886/`
        """
        return 1 / self.dg(x)

    def ddx(self, x):
        """Evaluates the second derivative of the inverse mapping at x.
        For details refer to the reference material provided at:
        `http://www.math-principles.com/2014/03/second-derivative-problems-reciprocal.html`
        """
        return -self.ddg(x) / self.dg(x) ** 3

    # Standard implementations which might not be efficient
    def g_and_dg(self, x, g_out=None, dg_out=None):
        return self.g(x, g_out), self.dg(x, dg_out)

    def g_and_dg_and_ddg(self, x, g_out=None, dg_out=None, ddg_out=None):
        return self.g(x, g_out), self.dg(x, dg_out), self.ddg(x, ddg_out)


class Problem(Mapping):
    """
    This is the abstract implementation of a problem.
    """

    def __init__(self):
        super().__init__()
        self.name = 'Default'
        self.x_min, self.x_max = None, None
        self.x0 = None
        self.n, self.m = None, None

    @abstractmethod
    def g(self, x, out=None):
        ...

    @abstractmethod
    def dg(self, x, out=None):
        ...

    @abstractmethod
    def ddg(self, x, out=None):
        ...


class Subproblem(Problem):
    def __init__(self, approximation=Taylor1(), limits=Bounds(xmin=0, xmax=1)):
        super().__init__()
        self.approx = approximation
        self.set_limits(limits)
        self.lims = parse_to_list(limits)

    def set_limits(self, *limits):
        self.lims = parse_to_list(*limits)

    def add_limits(self, *limits):
        self.lims.extend(parse_to_list(*limits))

    def build(self, x, f, df, ddf=None):
        self.n, self.m = len(x), len(f) - 1

        # Update the approximation
        self.approx.update(x, f, df, ddf)

        # Update the local problem bounds
        self.x_min = np.full_like(x, -np.inf)
        self.x_max = np.full_like(x, +np.inf)

        # Enforce restriction on the possible step size within the subproblem.
        # The step is restricted by the chosen move limit strategy as well as
        # the feasible range of the intervening variables. First the move
        # limits are applied to constraint the step size.
        for ml in self.lims:
            ml.update(x, f, df, ddf)
            ml.clip(self.x_min)
            ml.clip(self.x_max)

        # Additional constraint on the step size by the feasible range of the
        # intervening variables. This prevents the subsolver to make an update
        # that causes the intervening variable to reach unreachable values,
        # e.g. cross the lower/upper bounds in the MMA asymptotes.
        self.approx.clip(self.x_min)
        self.approx.clip(self.x_max)

        assert np.isfinite(self.x_min).all() and np.isfinite(self.x_max).all(), \
            "The bounds must be finite. Use at least one move-limit or bound."

    def g(self, x, out=None):
        return self.approx.g(x)

    def dg(self, x, out=None):
        return self.approx.dg(x)

    def ddg(self, x, out=None):
        return self.approx.ddg(x)


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
        :param kwargs: Optionally get the 2nd-order sensitivity array
        :return: self: For method cascading
        """
        return self

    @abstractmethod
    def g(self, x, out=None):
        ...

    @abstractmethod
    def dg(self, x, out=None):
        ...

    @abstractmethod
    def ddg(self, x, out=None):
        ...

    @abstractmethod
    def clip(self, x):
        ...
