import numpy as np
from sao.move_limits.move_limit import MoveLimit


class AdaptiveMoveLimit(MoveLimit):
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
        :param dx: = x_max - x_min: Variable bound interval for relative step-size
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

    def update(self, x, f=None, df=None, ddf=None):
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
        # This only updates the `self.x_min` and `self.x_min` variables
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

        # apply the move limits to x_min and x_max
        self.x_min = self.x - self.factor * self.max_dx
        self.x_max = self.x + self.factor * self.max_dx
        return self
