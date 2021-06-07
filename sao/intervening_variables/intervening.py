import math

from abc import ABC, abstractmethod


class Intervening(ABC):
    """Abstract base class for the intervening variable mapping.

    This class provides a change of variables from y = f(x), transforming the
    variables x to y using a given transformation function f. Any child class
    should provide the functionality to compute the mapping y = f(x), as well
    as the first and second derivatives. Additionally, the inverse mapping
    should be provided, reversing the transformation.

    For details on the formulation, in specific regarding the first and
    second derivatives of the mapping and their inverses, see the reference
    material at: `ReferenceFiles/TaylorExpansion.pdf`.
    """

    @property
    def alpha(self):
        """A lower bound of the feasible range for the intervening variable."""
        return -math.inf

    @property
    def beta(self):
        """A upper bound of the feasible range for the intervening variable."""
        return +math.inf

    @abstractmethod
    def y(self, x):
        """Evaluates the mapping y = f(x)."""
        ...

    @abstractmethod
    def dydx(self, x):
        """Evaluates the first derivative of the mapping at x."""
        ...

    @abstractmethod
    def ddyddx(self, x):
        """Evaluates the second derivatives of the mapping at x."""
        ...

    def dxdy(self, x):
        """Evaluates the first derivative of the inverse mapping at x.

        For details refer to the reference material provided at:
        `ReferenceFiles/TaylorExpansion.pdf`
        """
        return 1 / self.dydx(x)

    def ddxddy(self, x):
        """Evaluates the second derivative of the inverse mapping at x.

        """
        return -self.ddyddx(x) / self.dydx(x) ** 3

    def update(self, *args, **kwargs):
        """Perform inplace updates of the state of the intervening variable.

        This allows to perform additional functionality to update the state
        of the intervening variable, for instance to keep track of information
        at previous iterations etc.
        """
        pass

    def clip(self, x):
        """Default clipping is none."""
        return x
