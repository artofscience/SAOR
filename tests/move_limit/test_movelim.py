import pytest
import numpy as np


from sao.move_limits.move_limit import GeneralMoveLimit, Bound, MoveLimit


def test_generalmovelimit():
    x = np.array([-1.0, -0.5, 1.0, 2.0, 0.75, 0.1, 1.13])

    ml = GeneralMoveLimit()
    x1 = x.copy()
    xcl = ml.clip(x1)
    assert x1 is xcl, "xcl must be a reference of x1"
    assert max(xcl) == max(x), "must be unclipped"
    assert min(xcl) == min(x), "must be unclipped"


def test_bound_uniform():
    x = np.array([-1.0, -0.5, 1.0, 2.0, 0.75, 0.1, 1.13])

    ml = Bound(0.0, 1.0)  # Clip between 0 and 1
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

    ml = Bound(xlow, xupp)  # Clip between the two vectors
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
    xcl = ml.clip(x+dx)
    assert (xcl == x+dx).all()

    ml.update(x)  # Now it activates the movelimit
    xcl = ml.clip(x+dx)
    assert max(xcl-x) <= 0.1
    assert min(xcl-x) >= -0.1
    assert np.allclose(xcl-x, np.array([0.1, 0.1, 0.01, 0.05, -0.1, -0.01, -0.05]))


def test_movelimit_uniform_relative():
    x    = np.array([-1.0, -0.5, 1.00,  2.00,  0.75,  0.10,  1.13])
    dx   = np.array([ 0.1,  0.2, 0.01,  0.05, -0.11, -0.01, -0.05])
    xlow = np.array([ 0.0, -0.1, 0.20, -0.30,  0.40, -0.50,  0.60])
    xupp = np.array([ 0.1,  0.2, 0.30,  0.40,  0.50,  0.60,  0.70])

    ml = MoveLimit(0.1, xupp-xlow).update(x)
    xcl = ml.clip(x+dx)
    assert np.allclose(xcl-x, np.array([0.01, 0.03, 0.01, 0.05, -0.01, -0.01, -0.01]))


def test_movelimit_vector_absolute():
    x    = np.array([-1.0, -0.5, 1.00, 2.00,  0.75,  0.10,  1.13])
    dx   = np.array([ 0.1,  0.2, 0.01, 0.05, -0.11, -0.01, -0.05])
    dxml = np.array([ 0.1, 0.01, 0.02, 0.04,  0.10,  0.10,  0.03])

    ml = MoveLimit(dxml).update(x)
    xcl = ml.clip(x+dx)
    assert np.allclose(xcl-x, np.array([0.1, 0.01, 0.01, 0.04, -0.1, -0.01, -0.03]))


def test_movelimit_vector_relative():
    x    = np.array([-1.0, -0.5, 1.00, 2.00,  0.75,  0.10,  1.13])
    dx   = np.array([ 0.1,  0.2, 0.01, 0.05, -0.11, -0.01, -0.05])
    dxml = np.array([ 0.1, 0.01, 0.02, 0.04,  0.10,  0.10,  0.03])
    xlow = np.array([ 0.0, -0.1, 0.20, -0.30,  0.40, -0.50,  0.60])
    xupp = np.array([ 0.1,  0.2, 0.30,  0.40,  0.50,  0.60,  0.70])
    # dxre = np.array([0.01, 0.003, 0.002, 0.028, 0.01, 0.11, 0.003])  # Allowable movelimit

    ml = MoveLimit(dxml, xupp-xlow).update(x)
    xcl = ml.clip(x+dx)
    assert np.allclose(xcl-x, np.array([0.01, 0.003, 0.002, 0.028, -0.01, -0.01, -0.003]))


if __name__ == '__main__':
    test_generalmovelimit()
    test_bound_uniform()
    test_bound_vector()
    test_movelimit_uniform_absolute()
    test_movelimit_uniform_relative()
    test_movelimit_vector_absolute()
    test_movelimit_vector_relative()

