import numpy as np


class MoveLimit:
    """
    The simplest move limit strategy that always clips the global bounds of
    the problem, i.e. no restrictions are made on the possible steps size.
    The move limits are evaluated by calling an instance of a move limit class,
    i.e. the subclasses of ``MoveLimit``, where ``__call__`` or ``clip`` clips the
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

    def update(self, x, f, df, ddf=None):
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


class TrustRegion(MoveLimit):
    """
    The move limit constrains the allowed step size to be within a trust region
    around the current point ``x``. The size of the trusted region is expressed
    as a factor of the total range of the variable, i.e.
    ``move_limit * (x_max - xmin)``. This region is then used to truncated the
    step size to obtain the corresponding lower and upper bounds of the
    variables.
    """

    def __init__(self, move_limit=0.1, dx=1.0):
        """
        :param move_limit: The absolute move limit, in case dx is not given, or relative to dx
        :param dx: = x_max - xmin: Variable bound interval for relative step-size
        """
        super().__init__()

        """Stores the desired step-size (``trust region``)."""
        self.max_dx = abs(move_limit) * dx

    def update(self, x, f, df, ddf=None):
        self.x_min = x - self.max_dx
        self.x_max = x + self.max_dx
        return self


class MoveLimitAdaptive(TrustRegion):
    """
    This provides an adaptive move limit strategy as originally proposed
    within the MMA algorithm. It is essentially the asymptote update rule,
    which aims to reduce oscillations between iterations (if observed) and
    adepts the allowed step-size for the variables accordingly.

    The maximum allowable move is determined by
    `max_move = factor * move_limit`,
    where the factor is updated according to any detected oscillations
    """
    def __init__(self,
                 move_limit=0.1,
                 dx=1.0,
                 ml_init=0.5,
                 ml_incr=1.2,
                 ml_decr=0.7,
                 ml_bound=0.01,
                 oscillation_tol=1e-10):
        """
        :param move_limit: Relative/absolute move limit, depending if dx is given or not
        :param dx: = x_max - xmin: Variable bound interval for relative step-size
        :param ml_init: Initial factor of move limit (max move limit = factor * move limit)
        :param ml_incr: Increase factor for non-oscillations
        :param ml_decr: Decrease factor for oscillations
        :param ml_bound: Minimum factor
        :param oscillation_tol: Tolerance for detecting oscillations
        """
        super().__init__(move_limit=move_limit, dx=dx)
        self.ml_init = abs(ml_init)
        self.ml_incr = abs(ml_incr)
        self.ml_decr = abs(ml_decr)
        self.ml_bound = abs(ml_bound)
        self.osc_tol = abs(oscillation_tol)
        self.factor = None

        # history variables
        self.x, self.x_old1, self.x_old2 = None, None, None

    def update(self, x, f, df, ddf=None):
        """
        This method updates the allowable move limits from the current point.
        It has a similar structure to the asymptote update rule given by Svanberg1998

        :param x: Design variable vector of size [n]
        :return: self
        """
        # update stored variables from previous iterations
        self.x_old2, self.x_old1, self.x = self.x_old1, self.x, x.copy()

        if self.factor is None:
            self.factor = self.ml_init * np.ones_like(x)

        # If x_old2 is None, not enough iterations were performed
        # to have all (required) history information available.
        # This only updates the `self.xmin` and `self.xmin` variables
        # using a default move limit approach
        if self.x_old2 is not None:
            # To test if a design variable oscillates between iterations, we
            # evaluate the product of its change between the last two iterations.
            # This product will always be positive when the change in the variable
            # had the same "direction" between both iterations. So, when a negative
            # result is observed for a variable, that variable must have been
            # oscillating.
            oscillates = (self.x - self.x_old1) * (self.x_old1 - self.x_old2) / self.max_dx

            # To reduce the oscillations, the oscillating variables are assigned
            # `ml_decr` compared to `ml_incr` for the non oscillating variables.
            self.factor[oscillates > self.osc_tol] *= self.ml_incr
            self.factor[oscillates < self.osc_tol] *= self.ml_decr

            # Clip the factor between minimum factor and 1.0
            # (steps > move_limit are not allowed)
            np.clip(self.factor, self.ml_bound, 1.0, out=self.factor)

        # apply the move limits to xmin and x_max
        self.x_min = self.x - self.factor * self.max_dx
        self.x_max = self.x + self.factor * self.max_dx
        return self
