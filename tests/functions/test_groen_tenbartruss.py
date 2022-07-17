#
import pytest
#
from examples.groen.tenbartruss import tenbartruss_t2r
#
def test_tenbartruss_t2r():
#
    f_r=[2098.2,2388.6,1630.4,1766.4,1587.2,1511.4,1506.6,1506.0,1505.8,1505.5,1505.2,1505.0,1504.7]
#
#   history = tenbartruss_t2r('pdip')
#
#   assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
#   assert pytest.approx(history.getcol('inf'), abs=1e-2) == in_ref
#
    history = tenbartruss_t2r('t2dual')
#
    assert pytest.approx(history.getcol('f0'), rel=1e-3) == f_r
#
if __name__ == "__main__":
    test_tenbartruss_t2r()
