import numpy as np
from sao.move_limits.bounds import Bounds

class MoveLimit(Bounds):
    """
    The move limit constrains the allowed step size to be within a trust region
    around the current point ``x``. The size of the trusted region is expressed
    as a factor of the total range of the variable, i.e.
    ``move_limit * (x_max - x_min)``. This region is then used to truncated the
    step size to obtain the corresponding lower and upper bounds of the
    variables.
    """

    def __init__(self, move_limit=0.1, dx=1.0):
        """
        :param move_limit: The absolute move limit, in case dx is not given, or relative to dx
        :param dx: = x_max - x_min: Variable bound interval for relative step-size
        """
        super().__init__()

        """Stores the desired step-size (``trust region``)."""
        self.max_dx = abs(move_limit) * dx

    def update(self, x, f=None, df=None, ddf=None):
        self.x_min = x - self.max_dx
        self.x_max = x + self.max_dx
        return self

class MoveLimitFraction(Bounds):
    """
    The move limit strategy by (Svanberg 1987) of the 'Traditional method', that is
    x_min = x/fraction
    x_max = x*fraction, with fraction = 2.0
    """
    def __init__(self, fraction=2):
        self.fraction = fraction

    def update(self, x, f=None, df=None, ddf=None):
        self.x_min = x/self.fraction
        self.x_max = x*self.fraction
        return self


