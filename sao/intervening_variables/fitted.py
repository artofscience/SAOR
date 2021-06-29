from .intervening import Intervening


class ReciFit(Intervening):
    """Reciprocal like intervening variables y = 1 / (a * x + b).

    Coefficients -a_i- and -b_i- are calculated by forcing the current approximate response to
    satisfy the exact response value and its gradient at the previous point.
    """

    def __init__(self):
        """Initialise the reciprocal fitting function y_i = 1 / (a_i * x_i + b_i)."""
        self.x = None
        self.a, self.b = None, None
        self.f, self.df, self.dfold1, self.fold1 = None, None, None, None

    def update(self, x, f, df, *args, **kwargs):
        self.fold1 = self.f
        self.f = f
        self.dfold1 = self.df
        self.df = df
        self.x = x
        self.get_coefficients()

    def get_coefficients(self):
        """
        Calculates the coefficients -a_i- and -b_i- by fitting the current approximate response
        through the previous point and matching the previous point gradient.
        :return:
        """
        if self.dfold1 is not None:
            self.a = - 0.5 * (self.df + self.dfold1) / (0.5 * (self.f + self.fold1)) ** 2
            self.b = 2 / (self.f + self.fold1) - self.a * self.x
        else:
            self.a = - self.df / self.f ** 2
            self.b = 1 / self.f - self.a * self.x
        self.a[(abs(self.a) < 1e-2) * (self.a >= 0)] = 1e-2
        self.a[(abs(self.a) < 1e-2) * (self.a < 0)] = -1e-2

    def y(self, x):
        y = 1 / (self.a * x + self.b)
        return y

    def dydx(self, x):
        dydx = - self.a / (self.a * x + self.b) ** 2
        return dydx

    def ddyddx(self, x):
        ddyddx = 2 * self.a ** 2 / (self.a * x + self.b) ** 3
        return ddyddx
