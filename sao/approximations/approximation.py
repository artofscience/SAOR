from abc import ABC, abstractmethod


class Approximation(ABC):
    def __init__(self):
        self.x, self.y = None, None
        self.f, self.df, self.ddf = None, None, None
        self.P, self.Q = None, None

        self.m = -1  # Number of constraints
        self.n = -1  # Number of variables

    def generate(self, y, f, P, Q=None):
        """ Puts in data from the original problem. Once per design iteration.

        :param y:
        :param f:
        :param P:
        :param Q:
        :return:
        """
        self.y = y
        self.f, self.P, self.Q = f, P, Q

        self.m = len(self.f) - 1
        self.n = len(self.y)

        msg = (f'Expect sensitivity of size {self.m+1}x{self.n}: '
               f'Received {self.P.shape}.')
        assert self.P.shape == (self.m + 1, self.n), msg

        if self.Q is not None:
            msg = (f"Expected ddf size: {self.m+1}x{self.n}: "
                   f"Received: {self.Q.shape}.")
            assert self.Q.shape == (self.m + 1, self.n), msg

        return self

    @abstractmethod
    def g(self, y=None, dy=None, ddy=None):
        ...

    @abstractmethod
    def dg(self, y=None, dy=None, ddy=None):
        ...

    @abstractmethod
    def ddg(self, y=None, dy=None, ddy=None):
        ...
