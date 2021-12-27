import numpy as np

from .intervening import Intervening
from .exponential import Linear, Reciprocal
from .split import PositiveNegative


class ConLin(PositiveNegative):
    """The ``ConLin`` formulation of intervening variables.

    Introduced in: https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620230307.
    This is a 1st-order Taylor expansion with respect to the mixed
    (i.e. a combination of ``Linear`` and ``Reciprocal``)

    Intervening variables:

    .. math::
        y_i = x_i \\quad&\\text{if}\\quad dg_j/dx_i >= 0 \\\\
        y_i = \\frac{1}{x_i} \\quad&\\text{if}\\quad dg_j/dx_i < 0
    """
    def __init__(self):
        super().__init__(Linear(), Reciprocal())

    def clip(self, x):  # TODO Maybe add some tolerance like albefa to keep the value from 0
        return np.maximum(x, 0.0, out=x)
