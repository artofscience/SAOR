from sao.convergence_criteria.criteria import Criterion


class IterationCount(Criterion):
    """Enforces a maximum number of iterations.

    This keeps track of an internal counter that increments on each evaluation
    of ``__call__``. Thus, when setting up a loop as follows, this criterion
    will ensure at most 50 iterations are performed.

    >>> convergence = IterationCount(50):
    >>> while not convergence
    """

    def __init__(self, max_iteration=50):
        super().__init__()
        self.iteration = 0
        self.max_iteration = max_iteration

    def __call__(self):
        self.iteration += 1
        self.done = self.iteration >= self.max_iteration


class Feasibility(Criterion):
    """Enforces feasibility of all constraints with some "slack"."""

    def __init__(self, contraints, slack=1e-4):
        """Initialise the feasibility criteria with some allowed "slack".

        The slack variable can be provided as a single value or as a list of
        value, one per available constraint function. When only a single value
        is provided, that value is considered for all constraint functions.
        """
        super().__init__()
        self.constraints = contraints

        # Test if the provided slack variables allow iteration, if not, the
        # single value is repeated for the required number of constraints.
        try:
            _ = iter(slack)
            self.slack = slack
        except TypeError:
            self.slack = [slack] * len(self.constraints)

        err_msg = f"Wrong number of slack variables: {len(self.slack)}"
        assert len(self.slack) == len(self.constraints), err_msg

    def __call__(self):
        """Assert feasibility for each constraint with its allowed slack."""
        self.done = all(c < s
                        for (c, s) in zip(self.constraints, self.slack))
