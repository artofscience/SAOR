import numpy as np

from .intervening import Intervening


class PositiveNegative(Intervening):
    def __init__(self, left: Intervening, right: Intervening):
        self.left = left
        self.right = right
        self.positive = None

    def update(self, x, f, df, *args, **kwargs):
        self.positive = df >= 0

    def y(self, x):
        return np.where(self.positive, self.right.y(x), self.left.y(x))

    def dydx(self, x):
        return np.where(self.positive, self.right.dydx(x), self.left.dydx(x))

    def ddyddx(self, x):
        return np.where(self.positive, self.right.ddyddx(x), self.left.ddyddx(x))

    def clip(self, x):
        self.left.clip(x)
        self.right.clip(x)
        return x
