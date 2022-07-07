#
import pytest
#
from examples.groen.cantilever_t2r import cantilever_t2r
#
def test_cantilever_t2r():
#
    f0_ref = [1.560, 1.336, 1.323, 1.340, 1.340, 1.340]
    in_ref = [0.000, 0.10, 0.03, 0.00, 0.00, 0.00]
#
    history = cantilever_t2r('pdip')
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('inf'), abs=1e-2) == in_ref
#
    history = cantilever_t2r('t2dual')
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('inf'), abs=1e-2) == in_ref
#
if __name__ == "__main__":
    test_cantilever_t2r()
