from abc import ABC, abstractmethod


class Approximation(ABC):
    def __init__(self, **kwargs):
        self.y = None
        self.f, self.df, self.ddf = None, None, None
        self.dfdy, self.ddfddy = None, None

        self.m = -1  # Number of constraints
        self.n = -1  # Number of variables

        self.force_convex = kwargs.get('force_convex', True)

    def update(self, x, y, f, df, dxdy, **kwargs):
        """

        :param x: Current design
        :param y: A method that returns the intervening variables at the current design, i.e. y(x)
        :param f: A vector of size [m+1] that holds the response values at the current design -x-
        :param df: A matrix of size [m+1, n] that holds the sensitivity values at the current design -x-
        :param dxdy: A method that returns the derivative of the inverse intervening variable function, i.e. dx/dy(y(x))
        :param kwargs: Optionally get the 2nd-order sensitivity array
        :return:
        """
        self.y = y(x).T
        self.f, self.dfdy = f, df * dxdy(x)

        # 2nd-order part
        ddf = kwargs.get('ddf', None)
        if ddf is not None:
            ddxddy = kwargs.get('ddxddy', None)
            self.ddfddy = ddf * dxdy(x) ** 2 + df * ddxddy(x)

        self.m = len(self.f) - 1
        self.n = len(self.y)

        msg = (f'Expect sensitivity of size {self.m+1}x{self.n}: '
               f'Received {self.dfdy.shape}.')
        assert self.dfdy.shape == (self.m + 1, self.n), msg

        if self.ddfddy is not None:
            msg = (f"Expected ddf size: {self.m+1}x{self.n}: "
                   f"Received: {self.ddfddy.shape}.")
            assert self.ddfddy.shape == (self.m + 1, self.n), msg

            if self.force_convex:
                self.enforce_convexity()

        return self

    def enforce_convexity(self):
        # self.ddfddy.setflags(write=1)
        self.ddfddy[self.ddfddy < 0] = 0            # becomes read-only after the 1st iteration. Why?

    @abstractmethod
    def g(self, y):
        ...

    @abstractmethod
    def dg(self, y, dy):
        ...

    @abstractmethod
    def ddg(self, y, dy, ddy):
        ...
