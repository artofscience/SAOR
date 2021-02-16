from __future__ import annotations  # for type hinting to class itself
import abc


class Convergence(object):
    """An abstract convergence criteria."""
    def __init__(self):
        self._done = False

    def __and__(self, other) -> Convergence:
        """Combines two criteria using the `&` operator.

        >>> MaxIteration(20) & ObjectiveChange(1e-3)
        """
        return And(self, other)

    def __or__(self, other) -> Convergence:
        """Combines two criteria using the `|` operator.

        >>> MaxIteration(20) | ObjectiveChange(1e-3)
        """
        return Or(self, other)

    def __bool__(self) -> bool:
        """Return true if the criteria is satisfied."""
        return self._done

    @abc.abstractmethod
    def __call__(self, **kwargs) -> bool:
        """The update function to test convergence."""
        return self._done


class And(Convergence):
    """And combination for two convergence criteria."""
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = (a, b)

    def __call__(self, problem, **kwargs):
        """Evaluates both criteria and combines the result."""
        self._done = self.a(problem) & self.b(problem)
        return self._done


class Or(Convergence):
    """Or combination for two convergence criteria."""
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = (a, b)

    def __call__(self, problem, **kwargs):
        """Evaluates both criteria and combines the result."""
        self._done = self.a(problem) | self.b(problem)
        return self._done


class MaxIteration(Convergence):
    """Maximum iteration convergence criteria."""
    def __init__(self, max_iter):
        super().__init__()
        self.iter = 0
        self.max_iter = max_iter

    def __call__(self, problem, **kwargs):
        """Returns true once internal counter > maximum iterations."""
        self.iter += 1
        self._done = self.iter > self.max_iter
        return self._done


class ObjectiveChange(Convergence):
    """Change in objective convergence criteria."""
    def __init__(self, tolerance=1e-5):
        self.tolerance = tolerance
        self.old = 0

    def __call__(self, problem, **kwargs):
        """Returns true once change in objective smaller than tolerance."""
        change = abs(problem.g[0] - self.old)
        self.old = problem.g[0]
        self._done = change < self.tolerance
        return self._done
