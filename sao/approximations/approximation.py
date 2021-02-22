from abc import ABC, abstractmethod


class Approximation(ABC):
    def __init__(self):
        self.x = None

        # TODO: If these bounds refer to the local sub-problem bounds, I would rename them to alpha, beta just to
        #  stress the difference with the problem's bound constraints xmin, xmax and be aligned with the literature.
        self.xmin, self.xmax = None, None     # Bounds of the sub-problem
        self.f, self.df, self.ddf = None, None, None

        self.m = -1  # Number of constraints
        self.n = -1  # Number of variables

    # TODO: I would pass the bounds separately, even though they are always passed in pairs of {xmin, xmax}.
    #  Imo it is more user-friendly if you don't have to use xmin as bounds[0].
    def update_approximation(self, x, bounds, f, df, ddf=None):
        """ Puts in data from the original problem. Once per design iteration.

        :param x:
        :param xmin: Bounds of local problem, limited by global, and movelimit
        :param xmax:
        :param f:
        :param df:
        :param ddf:
        :return:
        """
        self.x = x
        self.alpha, self.beta = bounds
        self.f, self.df, self.ddf = f, df, ddf

        # TODO: Do we need to calculate -n- and -m- every design iteration? (e.g. for adding/removing constraints)
        self.m = len(self.f) - 1
        self.n = len(self.x)

        msg = (f'Expect sensitivity of size {self.m+1}x{self.n}: '
               f'Received {self.df.shape}.')
        assert self.df.shape == (self.m + 1, self.n), msg

        if self.ddf is not None:
            msg = (f"Expected ddf size: {self.m+1}x{self.n}: "
                   f"Received: {self.ddf.shape}.")
            assert self.ddf.shape == (self.m + 1, self.n), msg

        self.build_approximation()
        return self

    def build_approximation(self):
        pass

    @abstractmethod
    def g_approx(self, x):
        ...

    @abstractmethod
    def dg_approx(self, x):
        ...

    @abstractmethod
    def ddg_approx(self, x):
        ...


