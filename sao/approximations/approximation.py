from abc import ABC, abstractmethod


class Approximation(ABC):

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

    # Standard implementations which might not be efficient
    def g_and_dg(self, x, g_out=None, dg_out=None):
        return self.g(x, g_out), self.dg(x, dg_out)

    def g_and_dg_and_ddg(self, x, g_out=None, dg_out=None, ddg_out=None):
        return self.g(x, g_out), self.dg(x, dg_out), self.ddg(x, ddg_out)

    def clip(self, x):
        return x
