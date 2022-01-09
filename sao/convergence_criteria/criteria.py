"""Abstract implementation for convergence criteria."""
import abc
import math
import operator

import numpy as np


class Criterion(abc.ABC):
    """Abstract base class for a boolean (convergence) criterion.

    The criteria evaluates it current status upon conversion towards a boolean
    value, i.e. when the ``__bool__`` function is invoked. This then evaluates
    the ``__call__`` method, which should be provided by the user. Herein, the
    ``self.done`` attribute should be set according to the current status of
    the desired criterion.

    By invoking ``__call__`` on conversion to bool, it allows to write while
    loops as follows:

    >>> convergence = ObjectiveChange()
    >>> while not convergence:
    >>> ...

    This base class also provides implementations for ``__and__`` and
    ``__or__`` that allows for easy composition of various child classes of
    ``Criterion``. These magic functions return an instance of ``Criteria``
    that evaluates the corresponding ``and`` or ``or`` operation on the
    combined criterion classes. For instance, to enforce either a combination
    of objective change and feasibility, or a total number of iterations:

    >>> objective_change = ObjectiveChange(tolerance=1e-4)
    >>> feasibility = Feasibility(slack=1e-4)
    >>> max_iter = IterationCount(500)
    >>> convergence = (objective_change & feasibility) | max_iter

    Finally, the base class implements ``__invert__``, such that the criteria
    can be flipped. For instance, if you want at least a number of iterations
    to be performed before convergence might be triggered:

    >>> convergence = objective_change & feasibility & ~max_iter

    """

    def __init__(self):
        """On initialisation the criterion is set to False."""
        self.done = False

    def __bool__(self):
        """Evaluates its criterion function and updates it status."""
        self.__call__()
        return self.done

    def __and__(self, other):
        """Return a combined ``Criteria`` from the ``and (&)`` operation."""
        return Criteria(self, other, operator.__and__)

    def __or__(self, other):
        """Return a combined ``Criteria`` from the ``or (|)`` operation."""
        return Criteria(self, other, operator.__or__)

    def __invert__(self):
        """Returns a combined ``Criteria`` with a flipped result.

        The boolean value is inverted by evaluating a "not equal", i.e.
        ``__ne__``, with respect to ``True``, causing the original value to be
        flipped in the returned, combined ``Criteria`` class.
        """
        return Criteria(self, lambda: True, operator.__ne__)

    @abc.abstractmethod
    def __call__(self):
        """User provided function to set the converged statue: ``self.done``"""
        ...


class Criteria(Criterion):
    """A boolean combination of two ``Criterion`` instances.

    This class keeps track of two criteria, e.g. "left" and "right". These are
    combined given the provided operator, typically ``__and__`` or ``__or__``.
    This class simplifies chaining of various boolean operations with multiple
    (sub)classes from ``Criterion``.
    """

    def __init__(self, left, right, op):
        super().__init__()
        self.left, self.right = left, right
        self.operator = op

    def __call__(self):
        """Ensure both criteria are called when called."""
        self.left()
        self.right()

    def __bool__(self):
        """Overloads ``bool`` to combine both criteria."""
        return self.operator(bool(self.left), bool(self.right))


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
