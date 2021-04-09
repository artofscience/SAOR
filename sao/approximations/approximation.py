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
        """ Puts in data from the original problem. Once per design iteration.

        :param y:
        :param f:
        :param dfdy:
        :param ddfddy:
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
                self.ddfddy = self.enforce_convexity(self.ddfddy)

        return self

    @staticmethod
    def enforce_convexity(matrix):
        matrix[matrix < 0] = 0
        return matrix

    @abstractmethod
    def g(self, y):
        ...

    @abstractmethod
    def dg(self, y, dy):
        ...

    @abstractmethod
    def ddg(self, y, dy, ddy):
        ...
