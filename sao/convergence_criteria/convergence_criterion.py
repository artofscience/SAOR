"""Abstract implementation for convergence criteria."""
import abc


# Abstract class for convergence criteria. All criteria inherit from this class
class ConvergenceCriterion(abc.ABC):
    """Abstract implementation for the convergence criteria.

    Each criteria should implement the ``__call__`` method that assigns the
    values to ``self.converged`` to indicate if the criteria is satisfied.
    """
    def __init__(self, variable, target=1e-4, ):
        """Initialise the criteria with variable name and target value.

        The variable name corresponds to the variable extracted from
        ``kwargs``, e.g. when set to ``self.variable = "x"`` the criteria
        will extract ``x`` form the provided keyword arguments.

        Typically the criteria is satisfied when a target is achieved, e.g. the
        current value is larger, equal, or bigger than the target.
        """
        self.variable = variable
        self.target = target
        self._done = False

    @property
    def converged(self) -> bool:
        """Returns the status of the criteria."""
        return self._done

    @converged.setter
    def converged(self, done: bool):
        """Setter for the ``converged`` status of the criteria."""
        self._done = done

    def get_variable(self, **kwargs):
        """Returns the current ``self.variable`` from ``kwargs``.

        An ``AssertionError`` is raised in case of missing arguments. In those
        situations the convergence criteria cannot be properly assessed."""

        err = f"Variable '{self.variable}' was expected for {type(self)}."
        assert self.variable in kwargs, err
        return kwargs.get(self.variable)

    @abc.abstractmethod
    def __call__(self, **kwargs):
        """Abstract method to evaluate the current criteria."""
        ...


class AnyCriteria(ConvergenceCriterion):
    """Converges as soon as one of its criteria is reached."""
    def __init__(self, criteria):
        super().__init__(variable=None, target=None)

        # ensure the criteria collection supports iteration
        try:
            _ = iter(criteria)
            self.critera = criteria
        except TypeError:
            self.criteria = [criteria]

    def __call__(self, **kwargs):
        for criterion in self.critera:
            criterion(**kwargs)

        self.converged = any(c.converged for c in self.criteria)


class AllCriteria(AnyCriteria):
    """Converges only once *all* of its criteria is reached."""
    def __call__(self, **kwargs):
        for criterion in self.critera:
            criterion(**kwargs)

        self.converged = all(c.converged for c in self.criteria)
