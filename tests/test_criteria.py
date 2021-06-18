import pytest
import numpy as np

from sao.convergence_criteria.criteria import Criterion
from sao.convergence_criteria.criteria import IterationCount
from sao.convergence_criteria.criteria import ObjectiveChange
from sao.convergence_criteria.criteria import VariableChange
from sao.convergence_criteria.criteria import Feasibility


class Counter(Criterion):
    """A criteria class for testing purposes.

    The criteria is satisfied once a number of calls are performed.
    """
    def __init__(self, target):
        super().__init__()
        self.count = 0
        self.target = target

    def __call__(self):
        self.count += 1
        self.done = self.count >= self.target


def test_counter():
    """Ensure criteria is satisfied after hitting its target."""
    counter = Counter(1)
    assert counter


def test_and_counters():
    """Only returns ``true`` once all criteria are satisfied."""
    counter = Counter(1) & Counter(2) & Counter(3)
    assert not counter
    assert not counter
    assert counter


def test_or_counters():
    """Returns true once any of the criteria is satisfied."""
    counter = Counter(2) | Counter(3) | Counter(4)
    assert not counter
    assert counter


def test_not_counter():
    """Assert that criteria can be inverted using ``~``."""
    counter = ~Counter(2)
    assert counter
    assert not counter


@pytest.mark.parametrize('n', [1, 5])
def test_iteration_count(n):
    criteria = IterationCount(n)
    for _ in range(n):
        assert not criteria.done, "Should not be done yet..."
        criteria()
    assert criteria.done, "Should be marked completed."


@pytest.mark.parametrize('normalise', [True, False])
def test_objective_change(normalise):
    objective = np.array([1.0])
    criteria = ObjectiveChange(objective, normalise=normalise)

    # Do not mark as complete the first call
    criteria()
    assert not criteria.done

    # Mark as complete invoking with the same value
    criteria()
    assert criteria.done

    # Mark not as complete when some difference > tolerance is present
    objective += 2 * criteria.tolerance
    criteria()
    assert not criteria.done, f'{criteria.objective}, {criteria.previous}'


def test_variable_change():
    variables = np.array(range(10), dtype=float)
    criteria = VariableChange(variables)

    # Do not mark as complete the first call
    criteria()
    assert not criteria.done

    # Mark complete using the same values
    criteria()
    assert criteria.done

    # Fail when a single variable change is not sufficient
    variables[-1] += 2 * criteria.tolerance
    criteria()
    assert not criteria.done


def test_feasibility_change():
    constraints = np.ones(10, dtype=float)
    criteria = Feasibility(constraints)

    # Fail when values are positive
    criteria()
    assert not criteria.done

    # Succeed when positive, but within slack variables
    constraints *= np.array(criteria.slack)/2
    criteria()
    assert criteria.done

    # Succeed when all are negative
    constraints *= -1
    criteria()
    assert criteria.done
