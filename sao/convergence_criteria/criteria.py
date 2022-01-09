"""Abstract implementation for convergence criteria."""
import abc
import operator


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


