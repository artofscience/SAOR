import copy
import math
import numpy as np


class NoMoveLimit(object):
    """
    The simplest move limit strategy that always returns the global bounds of
    the problem, i.e. no restrictions are made on the possible steps size.

    The move limits are evaluated by calling an instance of a move limit class,
    i.e. the subclasses of ``NoMoveLimit``, where ``__call__`` return the
    corresponding lower and upper bounds.
    """
    def __init__(self, xmin, xmax):
        """Store the global problem bounds on initialisation."""
        self.xmin = xmin
        self.xmax = xmax
        self.dx = self.xmax - self.xmin

    def __call__(self, x):
        """Returns the global bounds of the problem."""
        return self.xmin, self.xmax


class MoveLimit(NoMoveLimit):
    """
    The move limit constrains the allowed step size to be within a trust region
    around the current point ``x``. The size of the trusted region is expressed
    as a factor of the total range of the variable, i.e.
    ``move_limit * (xmax - xmin)``. This region is then used to truncated the
    step size to obtain the corresponding lower and upper bounds of the
    variables.
    """
    def __init__(self, xmin=-math.inf, xmax=math.inf, move_limit=0.1):
        """Stores the desired step-size (``trust region``)."""
        super().__init__(xmin, xmax)
        self.move_limit = abs(move_limit)

        # The move-limit is interpreted as a factor on the global bounds of
        # each design variable, and thus cannot exceed [0, 1].
        assert 0 <= self.move_limit <= 1.0, \
            f"{self.move_limit=} cannot exceed [0, 1]."

    def __call__(self, x, factor=None):
        """Obtain the lower and upper bounds around the current point``x``.

        An optional argument ``factor`` can be passed in to overwrite the
        stored attribute ``self.move_limit``. This can be used to reuse the
        application of the move limit with various sizes of the trusted region.
        """
        # We take the absolute value here to prevent swapping the direction
        # when a negative value might be supplied.
        factor = abs(factor) if factor is not None else self.move_limit

        alpha = np.maximum(self.xmin, x - factor * self.dx)
        beta = np.minimum(self.xmax, x + factor * self.dx)
        return alpha, beta


class MoveLimitMMA(MoveLimit):
    """
    This provides and adaptive move limit strategy as originally proposed
    within the MMA algorithm. It is essentially the asymptote update rule,
    which aims to reduce oscillations between iterations (if observed) and
    adepts the allowed step-size for the variables accordingly.
    """
    def __init__(self,
                 xmin=-math.inf,
                 xmax=math.inf,
                 move_limit=0.1,
                 ml_init=0.5,
                 ml_incr=1.2,
                 ml_decr=0.7,
                 ml_bound=10.0,
                 ml_albefa=0.1):
        super().__init__(xmin=xmin, xmax=xmax, move_limit=move_limit)
        # TODO: document all possible settings below
        self.ml_init = ml_init
        self.ml_incr = ml_incr
        self.ml_decr = ml_decr
        self.ml_bound = ml_bound
        self.ml_albefa = ml_albefa
        self.factor = self.ml_init * np.ones(len(xmin))

        # history variables
        self.x, self.xold1, self.xold2 = None, None, None
        self.low, self.upp = None, None

    def __call__(self, x):
        """
        This method updates the allowable move limits from the current point.
        It has a similar structure to the asymptote update rule given by:
        http://www.ingveh.ulg.ac.be/uploads/education/meca-0027-1/MMA_DCAMM_1998.pdf

        :param x: Design variable vector of size [n]
        :param kwargs:
        :return: alpha, beta: lower and upper move limits
        """

        # update stored variables from previous iterations
        self.xold2, self.xold1, self.x = self.xold1, self.x, x

        # start the bounds using the default move limit approach
        alpha, beta = super().__call__(x)

        if self.xold2 is None:
            # Not enough iterations were performed to have all (required)
            # history information available. This only updates the `self.low`
            # and `self.upp` variables using a default move limit approach
            self.low, self.upp = super().__call__(self.x, factor=self.factor)
            return alpha, beta

        # To test if a design variable oscillates between iterations, we
        # evaluate the product of its change between the last two iterations.
        # This product will always be positive when the change in the variable
        # had the same "direction" between both iterations. So, when a negative
        # result is observed for a variable, that variable must have been
        # oscillating.
        oscillates = (self.x - self.xold1) * (self.xold1 - self.xold2) <= 0

        # To reduce the oscillations, the oscillating variables are assigned
        # `asydecr` compared to `asyincr` for the non oscillating variables.
        self.factor[oscillates] = self.ml_incr
        self.factor[~oscillates] = self.ml_decr

        # update lower and upper asymptotes
        self.low = self.x - self.factor * (self.xold1 - self.low)
        self.upp = self.x + self.factor * (self.upp - self.xold1)

        # check min and max bounds of asymptotes, as they cannot be too close
        # or far from the variable (TODO: redundant?)
        lowmin = self.x - self.ml_bound * self.dx
        lowmax = self.x - 1 / (self.ml_bound**2) * self.dx
        uppmin = self.x + 1 / (self.ml_bound**2) * self.dx
        uppmax = self.x + self.ml_bound * self.dx

        # if given asymptotes cross boundaries put them to their max/min values
        # (TODO: redundant?)
        self.low = np.clip(self.low, lowmin, lowmax)
        self.upp = np.clip(self.upp, uppmin, uppmax)

        # move limits derived from the current asymptotes
        lower = self.low + self.ml_albefa * (self.x - self.low)
        upper = self.upp - self.ml_albefa * (self.upp - self.x)

        # apply the move limits to alpha and beta to obtain the most
        # conservative move limits, this effectively combines the standard move
        # limit strategy given by `self.move_limit` with the move limits
        # proposed by the intervening variables (asymptotes).
        alpha = np.maximum(alpha, lower)
        beta = np.minimum(beta, upper)

        return alpha, beta
