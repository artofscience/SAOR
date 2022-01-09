import numpy as np


class Bounds:
    """
    The simplest move limit strategy that always clips the global bounds of
    the problem, i.e. no restrictions are made on the possible steps size.
    The move limits are evaluated by calling an instance of a move limit class,
    i.e. the subclasses of ``Bounds``, where ``__call__`` or ``clip`` clips the
    given vector within the appropriate limits.
    """

    def __init__(self, xmin=-np.inf, xmax=np.inf):
        """
        Setup the object with a minimum and maximum global bound
        :param xmin: Minimum value
        :param xmax: Maximum value
        """
        self.x_min = xmin
        self.x_max = xmax

    def update(self, x, f=None, df=None, ddf=None):
        """
        Update function
        :param x: Current design vector
        :return: self
        """
        return self

    def __call__(self, x):
        """Clips the vector to the bounds of the move limit"""
        return self.clip(x)

    def clip(self, x):
        """
        Clips the given vector within the bounds of x_min and x_max, as stored in the object.
        This function changes the input vector, so it does not make a copy.
        :param x: The vector to be clipped
        :return: The clipped vector
        """
        return np.clip(x, self.x_min, self.x_max, out=x)

    def set_bounds(self, xmin=0.0, xmax=1.0):
        self.x_min = xmin
        self.x_max = xmax
