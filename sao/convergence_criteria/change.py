class ObjectiveChange(Criterion):
    """Keeps track of the relative objective changes between iterations."""

    def __init__(self, objective, tolerance=1e-4, normalise=False):
        super().__init__()
        self.objective = objective
        self.tolerance = tolerance
        self.previous = np.full_like(self.objective, np.inf)
        self.normalise = normalise
        self.value = None

    def __call__(self):
        """Evaluate the objective changes between iterations."""
        current = self.objective

        self.value = abs(current - self.previous)
        if self.normalise:
            # If previous is `np.inf`, the change is already set to infinite,
            # and the normalisation can be skipped to avoid division by `Inf`.
            if not np.isinf(self.previous):
                self.value /= abs(self.previous)

        self.done = bool(self.value < self.tolerance)

        # make sure to keep track of the previous iterations
        self.previous = current.copy()


class VariableChange(Criterion):
    """Enforces an absolute, maximum variable change on all design variables.

    The variable change is determined between two consecutive iterations and
    the tolerance should be achieved for all defined design variables. If
    desired, the variable change can be scaled, for instance to normalise the
    change with respect to the maximum range of the design variables.
    """

    def __init__(self, variables, tolerance=1e-4, scaling=1.0):
        """Initialise the criteria with a tolerance and scaling"""
        super().__init__()
        self.variables = variables
        self.tolerance = tolerance
        self.scaling = scaling
        self.previous = math.inf

    def __call__(self):
        """Assert all variables show a sufficiently small change."""
        current = self.variables

        # The (scaled) change of all variables should be sufficiently small
        # before the variable change is satisfied.
        self.done = all(
            abs((current - self.previous) / self.scaling) < self.tolerance)

        # keep track of the previous iterations value of the variables
        self.previous = current.copy()


class VariableChangeNorm(Criterion):
    """Enforces an absolute, maximum variable change on all design variables.

    The variable change is determined between two consecutive iterations and
    the tolerance should be achieved for all defined design variables. If
    desired, the variable change can be scaled, for instance to normalise the
    change with respect to the maximum range of the design variables.
    """

    def __init__(self, variables, tolerance=1e-4, scaling=1.0):
        """Initialise the criteria with a tolerance and scaling"""
        super().__init__()
        self.variables = variables
        self.tolerance = tolerance
        self.scaling = scaling
        self.previous = math.inf
        self.value = None

    def __call__(self):
        """Assert the norm of the variables' change is sufficiently small."""
        current = self.variables

        # The (scaled) change of all variables should be sufficiently small
        # before the variable change is satisfied.
        self.value = np.linalg.norm((current - self.previous) / self.scaling)
        self.done = (self.value < self.tolerance).astype(bool)

        # keep track of the previous iterations value of the variables
        self.previous = current.copy()
