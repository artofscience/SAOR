"""A collection of intervening variables exploiting fitting to information
aggregated across previous iterations."""

import numpy as np
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

    def update(self, x, f, df):
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
        # self.a[abs(self.a) < 1e-5] = 1e-5 * np.sign(self.a[abs(self.a) < 1e-5])

    def y(self, x):
        y = 1 / (self.a * x + self.b)
        return y

    def dydx(self, x):
        dydx = - self.a / (self.a * x + self.b) ** 2
        return dydx

    def ddyddx(self, x):
        ddyddx = 2 * self.a ** 2 / (self.a * x + self.b) ** 3
        return ddyddx

    def dxdy(self, x):
        dxdy = - (self.a * x + self.b) ** 2 / self.a
        return dxdy

    def ddxddy(self, x):
        ddxddy = 2 * (self.a * self.x + self.b) ** 3 / self.a
        return ddxddy


# TODO: Doesn't work properly. Either fix or delete.
class Bezier(Intervening):
    def __init__(self):
        self.x = None
        self.a, self.b, self.c = None, None, None
        self.f, self.fold1, self.fold2 = None, None, None

    def update(self, x, f, df):
        self.x = x
        self.fold2 = self.fold1
        self.fold1 = self.f
        self.f = f
        self.get_coefficients()

    def get_coefficients(self):
        if self.fold2 is None:
            self.a = 2 * self.f * np.ones_like(self.x)
        else:
            self.a = self.fold2 * np.ones_like(self.x)
        if self.fold1 is None:
            self.b = 1.5 * self.f * np.ones_like(self.x)
        else:
            self.b = self.fold1 * np.ones_like(self.x)
        self.c = self.f * np.ones_like(self.x)

    def y(self, x):
        y = (1-x)**2*self.a + 2*(1-x)*x*self.b + x**2*self.c
        return y

    def dydx(self, x):
        dydx = -2*self.a*(1-x) - 4*self.b*x + 2*self.b + 2*self.c*x
        return dydx

    def ddyddx(self, x):
        ddyddx = 2 * (self.a - 2*self.b + self.c)
        return ddyddx

    def dxdy(self, x):
        dxdy = -1 / (2 * (-self.a*self.c + self.a*self.y(x) + self.b**2 - 2*self.b*self.y(x) + self.c*self.y(x))**(1/2))
        return dxdy

    def ddxddy(self, x):
        ddxddy = (-self.a + 2*self.b - self.c) / (4*(-self.a*self.c + self.a*self.y(x) + self.b**2 - 2*self.b*self.y(x) + self.c*self.y(x))**(3/2))
        return ddxddy


# TODO: Doesn't work properly. Either fix or delete.
class PolyFit(Intervening):
    def __init__(self):
        self.x, self.xold1 = None, None
        self.a, self.b, self.c = None, None, None
        self.f, self.fold1 = None, None
        self.df, self.dfold1 = None, None

    def update(self, x, f, df):
        if self.fold1 is None:
            self.fold1 = 2 * f
            self.dfold1 = 2 * df
            self.xold1 = 2 * x
        else:
            self.xold1 = self.x
            self.fold1 = self.f
            self.dfold1 = self.df
        self.x = x
        self.f = f
        self.df = df
        self.get_coefficients()

    def get_coefficients(self):
        # Polynomial satisfies: current point, current gradient, previous gradient
        self.a = np.zeros_like(self.x)
        bot = abs(self.x - self.xold1)
        bot2 = np.asarray(np.where(bot > 1e-4))
        bot2 = bot2[0, :]
        self.a[bot2] = (self.df[:, bot2] - self.dfold1[:, bot2]) / (2 * (self.x[bot2] - self.xold1[bot2]))
        self.a[self.a < 0] = 0
        self.b = self.df - 2 * self.a * self.x
        self.c = self.f - self.a * self.x ** 2 - self.b * self.x

    def y(self, x):
        y = self.a*x**2 + self.b*x + self.c
        return y

    def dydx(self, x):
        dydx = 2*self.a*x + self.b
        return dydx

    def ddyddx(self, x):
        ddyddx = 2*self.a
        return ddyddx

    def dxdy(self, x):
        # self.x1 = -self.b/(2*self.a) + (self.b**2 + 4*self.a*(self.y(x)-self.c))**(1/2) / (2*self.a)
        # self.x2 = -self.b/(2*self.a) + (self.b**2 + 4*self.a*(self.y(x)-self.c))**(1/2) / (2*self.a)
        # if 0 < self.x1 < 1:
        #     dxdy = (1 / (4*self.a*(self.y(x) - self.c) + self.b**2)**(1/2))
        # elif 0 < self.x2 < 1:
        dxdy = -1 / (4 * self.a * (self.y(x) - self.c) + self.b ** 2) ** (1 / 2)
        return dxdy

    def ddxddy(self, x):
        if 0 < self.x1 < 1:
            ddxddy = - 2*self.a / (4*self.a*(self.y(x) - self.c) + self.b**2)**(3/2)
        elif 0 < self.x2 < 1:
            ddxddy = 2 * self.a / (4 * self.a * (self.y(x) - self.c) + self.b ** 2) ** (3 / 2)
        return ddxddy
