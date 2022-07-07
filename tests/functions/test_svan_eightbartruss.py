#
import pytest
#
from examples.svan.eightbar_mma_dual import eightbar_mma_dual
#
def test_eightbar_mma_dual():
#
    f0_ref = [13.05, 12.10, 11.67, 11.65, 11.64, 11.62, 11.60, 11.56, 11.52, 11.47, 11.41, 11.36, 11.31, 11.24, 11.23]
#
    s=3./4.
    history = eightbar_mma_dual(s)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
#
    f0_ref = [13.05, 12.10, 11.67, 11.65, 11.63, 11.60, 11.53, 11.44, 11.35, 11.25, 11.23]
#
    s=1./2.
    history = eightbar_mma_dual(s)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
#
    f0_ref = [13.05, 12.10, 11.67, 11.65, 11.61, 11.52, 11.42, 11.28, 11.23]
#
    s=1./4.
    history = eightbar_mma_dual(s)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
#
if __name__ == "__main__":
    test_eightbar_mma_dual()
