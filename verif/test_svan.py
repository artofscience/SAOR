#
import pytest
#
from examples.tbar import two_bar_truss
#
def test_two_bar_truss():
#
    f0_ref = [1.68, 1.43, 1.37, 1.44, 1.47, 1.51]
    history = two_bar_truss()
#
    assert pytest.approx(history.getcol('f0'), rel=1e-2) == f0_ref
#
if __name__ == "__main__":
    test_tbar()
#
