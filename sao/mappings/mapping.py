from abc import ABC, abstractmethod
import numpy as np


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


class EmptyMap(Mapping, ABC):
    def g(self, x):
        return x

    def dg(self, x):
        return np.ones_like(x)

    def ddg(self, x):
        return np.zeros_like(x)

    def update(self, x, f, df, ddf=None):
        pass

    def clip(self, x):
        return x


class Approximation(EmptyMap, ABC):
    '''
    Approximation is a function mapping f: R^n -> R
    '''

    def __init__(self, mapping=EmptyMap()):
        self.map = mapping

    def update(self, x, f, df, ddf=None):
        """
        This method updates the approximation instance.

        :param x: Current design
        :param f: A vector of size [m+1] that holds the response values at the current design -x-
        :param df: A matrix of size [m+1, n] that holds the sensitivity values at the current design -x-
        :param ddf: Optionally get the 2nd-order sensitivity array
        :return: self: For method cascading
        """
        self.map.update(x, f, df, ddf=None)
        self._update(x, f, df, ddf=None)

    def clip(self, x):
        x = self.map.clip(x)
        return self._clip(x)

    def _update(self, x, f, df, ddf=None):
        pass

    def _clip(self, x):
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

    def __init__(self, mapping=EmptyMap()):
        super().__init__(mapping)

    @abstractmethod
    def _g(self, x):
        ...

    @abstractmethod
    def _dg(self, x):
        ...

    @abstractmethod
    def _ddg(self, x):
        ...

    def g(self, x):
        '''Chain rule'''
        return self._g(self.map.g(x))

    def dg(self, x):
        '''Chain rule first derivative'''
        return self._dg(self.map.g(x)) * self.map.dg(x)

    def ddg(self, x):
        '''Chain rule second derivative'''
        return self._ddg(x) * self.map.dg(x) ** 2 + self._dg(x) * self.map.ddg(x)


class Exponential(Intervening):
    """A generic exponential intervening variable y = x^p.

    The general case for an exponential intervening varaibles that can take
    on various forms depending on the chosen power. Note: the implementation
    does not support ``p = 0`` to avoid a zero devision in the derivatives.
    """

    def __init__(self, mapping=EmptyMap(), p=1, xlim=1e-10):
        super().__init__(mapping)
        """
        Initialise the exponential intervening variable with a power.
        :param p: The power
        :param xlim: Minimum x, in case of negative p, to prevent division by 0
        """
        assert p != 0, f"Invalid power x^{p}, will result in zero division."
        self.p = p
        self.xlim = xlim

    def _g(self, x):
        return x ** self.p

    def _dg(self, x):
        return self.p * x ** (self.p - 1)

    def _ddg(self, x):
        return self.p * (self.p - 1) * x ** (self.p - 2)

    def _clip(self, x):
        if self.p < 0:
            return np.maximum(x, self.xlim, out=x)
        return x


class Taylor1(Approximation):
    def __init__(self, mapping=Exponential(p=1)):
        super().__init__(mapping)
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

