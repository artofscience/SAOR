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
        `https://www.physicsforums.com/threads/is-the-derivative-equal-to-one-over-the-derivative-of-the-inverse.63886/`
        """
        return 1 / self.dydx(x)

    def ddxddy(self, x):
        """Evaluates the second derivative of the inverse mapping at x.
        For details refer to the reference material provided at:
        `http://www.math-principles.com/2014/03/second-derivative-problems-reciprocal.html`
        """
        return -self.ddyddx(x) / self.dydx(x) ** 3

    def update(self, *args, **kwargs):
        """Perform inplace updates of the state of the intervening variable.

        This allows to perform additional functionality to update the state
        of the intervening variable, for instance to keep track of information
        at previous iterations etc.
        """
        return self

    def clip(self, x):
        """Default clipping is none."""
        return x
