import pytest
import numpy as np
from sao.move_limits import Bounds, MoveLimit, AdaptiveMoveLimit
from sao.move_limits.mixed_move_limit import MixedMoveLimit


def test_move_limit():
    x = np.array([-1.0, -0.5, 1.0, 2.0, 0.75, 0.1, 1.13])

    ml = Bounds()
    x1 = x.copy()
    xcl = ml.clip(x1)
    assert x1 is xcl, "xcl must be a reference of x1"
    assert max(xcl) == max(x), "must be unclipped"
    assert min(xcl) == min(x), "must be unclipped"


def test_bound_uniform():
    x = np.array([-1.0, -0.5, 1.0, 2.0, 0.75, 0.1, 1.13])

    ml = Bounds(0.0, 1.0)  # Clip between 0 and 1
    x1 = x.copy()
    xcl = ml.clip(x1)
    assert x1 is xcl, "xcl must be a reference of x1"
    assert max(xcl) <= 1.0
    assert min(xcl) >= 0.0
    assert (xcl == np.array([0.0, 0.0, 1.0, 1.0, 0.75, 0.1, 1.0])).all()

    ml.set_bounds(0.4, 0.6)
    x2 = x.copy()
    ml.clip(x2)
    assert max(x2) <= 0.6
    assert min(x2) >= 0.4


def test_bound_vector():
    x = np.array([-1.0, -0.5, 1.0, 2.0, 0.75, 0.1, 1.13])
    xupp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    xlow = np.array([0.0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6])

    ml = Bounds(xlow, xupp)  # Clip between the two vectors
    x1 = x.copy()
    xcl = ml.clip(x1)
    assert x1 is xcl, "xcl must be a reference of x1"
    assert (xcl <= xupp).all()
    assert (xcl >= xlow).all()
    assert (xcl == np.array([0.0, -0.1, 0.3, 0.4, 0.5, 0.1, 0.7])).all()

    ml.set_bounds(0.4, 0.6)
    x2 = x.copy()
    ml.clip(x2)
    assert max(x2) <= 0.6
    assert min(x2) >= 0.4


def test_movelimit_uniform_absolute():
    x = np.array([-1.0, -0.5, 1.0, 2.0, 0.75, 0.1, 1.13])
    dx = np.array([0.1, 0.2, 0.01, 0.05, -0.11, -0.01, -0.05])

    ml = MoveLimit(0.1)  # Without update it acts unbounded
    xcl = ml.clip(x + dx)
    assert (xcl == x + dx).all()

    ml.update(x)  # Now it activates the movelimit
    xcl = ml.clip(x + dx)
    assert max(xcl - x) <= 0.1
    assert min(xcl - x) >= -0.1
    assert np.allclose(xcl - x, np.array([0.1, 0.1, 0.01, 0.05, -0.1, -0.01, -0.05]))


def test_movelimit_uniform_relative():
    x = np.array([-1.0, -0.5, 1.00, 2.00, 0.75, 0.10, 1.13])
    dx = np.array([0.1, 0.2, 0.01, 0.05, -0.11, -0.01, -0.05])
    x_low = np.array([0.0, -0.1, 0.20, -0.30, 0.40, -0.50, 0.60])
    x_upp = np.array([0.1, 0.2, 0.30, 0.40, 0.50, 0.60, 0.70])

    ml = MoveLimit(0.1, x_upp - x_low).update(x)
    xcl = ml.clip(x + dx)
    assert np.allclose(xcl - x, np.array([0.01, 0.03, 0.01, 0.05, -0.01, -0.01, -0.01]))


def test_movelimit_vector_absolute():
    x = np.array([-1.0, -0.5, 1.00, 2.00, 0.75, 0.10, 1.13])
    dx = np.array([0.1, 0.2, 0.01, 0.05, -0.11, -0.01, -0.05])
    dxml = np.array([0.1, 0.01, 0.02, 0.04, 0.10, 0.10, 0.03])

    ml = MoveLimit(dxml).update(x)
    xcl = ml.clip(x + dx)
    assert np.allclose(xcl - x, np.array([0.1, 0.01, 0.01, 0.04, -0.1, -0.01, -0.03]))


def test_movelimit_vector_relative():
    x = np.array([-1.0, -0.5, 1.00, 2.00, 0.75, 0.10, 1.13])
    dx = np.array([0.1, 0.2, 0.01, 0.05, -0.11, -0.01, -0.05])
    dxml = np.array([0.1, 0.01, 0.02, 0.04, 0.10, 0.10, 0.03])
    x_low = np.array([0.0, -0.1, 0.20, -0.30, 0.40, -0.50, 0.60])
    x_upp = np.array([0.1, 0.2, 0.30, 0.40, 0.50, 0.60, 0.70])

    ml = MoveLimit(dxml, x_upp - x_low).update(x)
    xcl = ml.clip(x + dx)
    assert np.allclose(xcl - x, np.array([0.01, 0.003, 0.002, 0.028, -0.01, -0.01, -0.003]))


def test_movelimit_adaptive():
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    dx1 = np.array([0.1, -0.05, 0.03, 0.5, 0.3, 0.01])
    dx2 = np.array([-0.1, -0.01, -0.05, 0.3, 0.1, -0.1])

    movelim = 0.1
    v_init, v_incr, v_decr, v_bound = 0.5, 1.5, 0.5, 0.01
    ml = AdaptiveMoveLimit(movelim, ml_init=v_init, ml_incr=v_incr, ml_decr=v_decr, ml_bound=v_bound)
    ml.update(x)
    # Use the initial move limit
    xcl = ml.clip(x + dx1)
    assert np.allclose(xcl - x, np.array([0.05, -0.05, 0.03, 0.05, 0.05, 0.01]))

    x += dx1
    ml.update(x)
    # Again use the initial move limit
    xcl = ml.clip(x + dx2)
    assert np.allclose(xcl - x, np.array([-0.05, -0.01, -0.05, 0.05, 0.05, -0.05]))

    x += dx2
    ml.update(x)
    # Some limits have gone smaller, others larger
    xcl = ml.clip(x + dx1)
    assert np.allclose(xcl - x, np.array([0.025, -0.05, 0.025, 0.075, 0.075, 0.01]))

    # n_saturate_upper = np.ceil(np.log10(1 / v_init) / np.log10(v_incr))
    # The factors reach the upper bound (1.0) after 2 oscillations
    x += dx1
    ml.update(x)
    xcl = ml.clip(x + dx2)
    assert xcl[3] - x[3] == pytest.approx(movelim)
    assert xcl[4] - x[4] == pytest.approx(movelim)

    n_saturate_lower = int(np.ceil(np.log10(v_bound / v_init) / np.log10(v_decr)))
    for i in range(n_saturate_lower - 2):
        dx = dx2 if i % 2 == 0 else dx1  # Alternate between adding dx2 and dx1
        x += dx
        ml.update(x)
    dx = dx2 if (i + 1) % 2 == 0 else dx1
    xcl = ml.clip(x + dx)
    assert abs(xcl[0] - x[0]) == pytest.approx(v_bound * movelim)
    assert abs(xcl[2] - x[2]) == pytest.approx(v_bound * movelim)
    assert abs(xcl[5] - x[5]) == pytest.approx(v_bound * movelim)

    # Move from the smallest step to the largest step by doing steps in a single direction
    n_small_to_large = int(np.ceil(np.log10(1.0 / v_bound) / np.log10(v_incr)))
    for i in range(n_small_to_large):
        dx = 0.1
        x += dx
        ml.update(x)
    xcl = ml.clip(x + 1.0)  # All should be clipped at movelimit
    assert min(xcl - x) == pytest.approx(movelim)

    # Move from the largest step to the smallest step by doing steps in a single direction
    n_large_to_small = int(np.ceil(np.log10(v_bound / 1.0) / np.log10(v_decr)))
    for i in range(n_large_to_small):
        dx = -0.1 if i % 2 == 0 else 0.1
        x += dx
        ml.update(x)
    xcl = ml.clip(x + 1.0)  # All should be clipped at movelimit
    assert max(xcl - x) == pytest.approx(v_bound * movelim)


# @pytest.mark.parametrize('n', [10])
def test_mixed_move_limit():
    n = 10
    mix = MixedMoveLimit(n, default=Bounds(0.3, 0.8))
    mix.add_move_limit(Bounds(0.2, 0.9), var=[0, 1, 2])
    mix.set_move_limit(Bounds(0.0, 0.0), var=[2])
    x = np.linspace(0, 1, 10)
    mix.clip(x)
    assert x[0] == x[1] == 0.3
    assert x[2] == 0
    assert x[9] == x[8] == 0.8


def test_mixed_trust_region():
    n = 10
    mix = MixedMoveLimit(n, default=Bounds(0.3, 0.8))
    mix.add_move_limit(MoveLimit(0.1), var=[0, 1, 2])
    mix.set_move_limit(Bounds(0.0, 0.0), var=[2])
    x = np.linspace(0, 1, 10)
    mix.update(x)
    y = mix.clip(x + 0.2)
    assert y[0] == x[0] + 0.1
    assert y[1] == x[1] + 0.1
    assert y[2] == 0
    assert y[9] == y[8] == y[7] == y[6] == 0.8


if __name__ == '__main__':
    test_move_limit()
    test_bound_uniform()
    test_bound_vector()
    test_movelimit_uniform_absolute()
    test_movelimit_uniform_relative()
    test_movelimit_vector_absolute()
    test_movelimit_vector_relative()
    test_movelimit_adaptive()
    test_mixed_move_limit()
    test_mixed_trust_region()
