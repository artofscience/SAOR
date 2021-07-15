from sao.convergence_criteria.criteria import Criterion


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
