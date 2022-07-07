#
import pytest
#
from examples.svan.cantilever_mma import cantilever_mma
#
def test_cantilever_mma():
#
    f0_beg_ref = [1.560, 1.274, 1.270, 1.304, 1.319, 1.329, 1.333, 1.336]
    in_beg_ref = [0.000, 0.35, 0.27, 0.14, 0.08, 0.04, 0.02, 0.01]
    f0_end_ref = [1.340, 1.340]
    in_end_ref = [0.002, 0.001]
#
    t=1./16.
    history = cantilever_mma(t)
#
    assert pytest.approx(history.getcol('f0')[:8], abs=1e-2) == f0_beg_ref
    assert pytest.approx(history.getcol('inf')[:8], abs=1e-2) == in_beg_ref
    assert pytest.approx(history.getcol('f0')[-2:], abs=1e-2) == f0_end_ref
    assert pytest.approx(history.getcol('inf')[-2:], abs=1e-2) == in_end_ref
#
    f0_ref = [1.560, 1.285, 1.307, 1.331, 1.337, 1.339, 1.340]
    in_ref = [0.000, 0.23, 0.11, 0.03, 0.008, 0.002, 0.001]
#
    t=1./8.
    history = cantilever_mma(t)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('inf'), abs=1e-2) == in_ref
#
    f0_ref = [1.560, 1.309, 1.335, 1.340]
    in_ref = [0.000, 0.100, 0.01, 0.0005]
#
    t=1./4.
    history = cantilever_mma(t)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('inf'), abs=1e-2) == in_ref
#
    f0_ref = [1.560, 1.327, 1.338, 1.340]
    in_ref = [0.000, 0.05, 0.004, 0.0001]
#
    t=1./3.
    history = cantilever_mma(t)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('inf'), abs=1e-2) == in_ref
#
    f0_ref = [1.560, 1.387, 1.346, 1.341]
    in_ref = [0.000, 0.000, 0.000, 0.000]
#
    t=1./2.
    history = cantilever_mma(t)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('inf'), abs=1e-2) == in_ref
#
    f0_ref = [1.560, 1.448, 1.386, 1.358, 1.347, 1.343, 1.341]
    in_ref = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
#
    t=2./3.
    history = cantilever_mma(t)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('inf'), abs=1e-2) == in_ref
#
    f0_ref = [1.560, 1.477, 1.418, 1.383, 1.363, 1.352, 1.346, 1.343, 1.342, 1.341]
    in_ref = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
#
    t=3./4.
    history = cantilever_mma(t)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('inf'), abs=1e-2) == in_ref
#
if __name__ == "__main__":
    test_cantilever_mma()
