from abc import ABC, abstractmethod


class Approximation(ABC):

    @abstractmethod
    def update(self, x, f, df, ddf=None):
        """
        This method updates the approximation instance.

        :param x: Current design
        :param y: A method that returns the intervening variables at the current design, i.e. y(x)
        :param f: A vector of size [m+1] that holds the response values at the current design -x-
        :param df: A matrix of size [m+1, n] that holds the sensitivity values at the current design -x-
        :param dxdy: A method that returns the derivative of the inverse intervening variable function, i.e. dx/dy(y(x))
        :param kwargs: Optionally get the 2nd-order sensitivity array
        :return: self: For method cascading
        """
        ...

    @abstractmethod
    def g(self, x, out=None):
        ...

    @abstractmethod
    def dg(self, x, out=None):
        ...

    @abstractmethod
    def ddg(self, x, out=None):
        ...

    # Standard implementations which might not be efficient
    def g_and_dg(self, x, g_out=None, dg_out=None):
        return self.g(x, g_out), self.dg(x, dg_out)

    def g_and_dg_and_ddg(self, x, g_out=None, dg_out=None, ddg_out=None):
        return self.g(x, g_out), self.dg(x, dg_out), self.ddg(x, ddg_out)

    def clip(self, x):
        return x
