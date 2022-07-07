#
import pytest
#
from problems.svanberg1987 import TwoBarTruss
from examples.svan.twobartruss_pdip import MMA, CON, LIN
from examples.svan.twobartruss_pdip import twobartruss_pdip
from examples.svan.twobartruss_dual import twobartruss_dual
#
def test_twobartruss_pdip():
#
    f0_ref = [1.68, 1.43, 1.37, 1.44, 1.47, 1.51]
    x1_ref = [1.5, 1.39, 1.22, 1.39, 1.37, 1.41]
    x2_ref = [0.5, 0.25, 0.50, 0.25, 0.38, 0.38]

    problem = TwoBarTruss()
    obj = MMA('Weight',problem.n)
    con1 = MMA('Stress 1',problem.n)
    con2 = MMA('Stress 2',problem.n)

    history = twobartruss_pdip(obj,con1,con2)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('x1'), abs=1e-2) == x1_ref
    assert pytest.approx(history.getcol('x2'), abs=1e-2) == x2_ref
#
    f0_ref = [1.68, 1.43, 1.49, 1.43, 1.49, 1.43, 1.49, 1.43]
    x1_ref = [1.5, 1.39, 1.33, 1.39, 1.33, 1.39, 1.33, 1.39]
    x2_ref = [0.5, 0.25, 0.50, 0.25, 0.5, 0.25, 0.5, 0.25]

    problem = TwoBarTruss()
    obj = CON('Weight',problem.n)
    con1 = CON('Stress 1',problem.n)
    con2 = CON('Stress 2',problem.n)

    history = twobartruss_pdip(obj,con1,con2)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('x1'), abs=1e-2) == x1_ref
    assert pytest.approx(history.getcol('x2'), abs=1e-2) == x2_ref
#
    f0_ref = [1.68, 1.42, 1.27, 1.38, 1.28, 1.38, 1.28, 1.38]
    x1_ref = [1.5, 1.38, 1.14, 1.34, 1.15, 1.34, 1.15, 1.34]
    x2_ref = [0.5, 0.25, 0.50, 0.25, 0.5, 0.25, 0.5, 0.25]

    problem = TwoBarTruss()
    obj = LIN('Weight',problem.n)
    con1 = LIN('Stress 1',problem.n)
    con2 = LIN('Stress 2',problem.n)

    history = twobartruss_pdip(obj,con1,con2)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('x1'), abs=1e-2) == x1_ref
    assert pytest.approx(history.getcol('x2'), abs=1e-2) == x2_ref
#
def test_twobartruss_dual():
#
    f0_ref = [1.68, 1.43, 1.37, 1.44, 1.47, 1.51]
    x1_ref = [1.5, 1.39, 1.22, 1.39, 1.37, 1.41]
    x2_ref = [0.5, 0.25, 0.50, 0.25, 0.38, 0.38]

    problem = TwoBarTruss()
    obj = MMA('Weight',problem.n)
    con1 = MMA('Stress 1',problem.n)
    con2 = MMA('Stress 2',problem.n)
    sub='allmmadual'
    history = twobartruss_dual(obj,con1,con2,sub)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('x1'), abs=1e-2) == x1_ref
    assert pytest.approx(history.getcol('x2'), abs=1e-2) == x2_ref
#
    f0_ref = [1.68, 1.43, 1.49, 1.43, 1.49, 1.43, 1.49, 1.43]
    x1_ref = [1.5, 1.39, 1.33, 1.39, 1.33, 1.39, 1.33, 1.39]
    x2_ref = [0.5, 0.25, 0.50, 0.25, 0.5, 0.25, 0.5, 0.25]

    problem = TwoBarTruss()
    obj = CON('Weight',problem.n)
    con1 = CON('Stress 1',problem.n)
    con2 = CON('Stress 2',problem.n)
    sub='allcondual'
    history = twobartruss_dual(obj,con1,con2,sub)
#
    assert pytest.approx(history.getcol('f0'), abs=1e-2) == f0_ref
    assert pytest.approx(history.getcol('x1'), abs=1e-2) == x1_ref
    assert pytest.approx(history.getcol('x2'), abs=1e-2) == x2_ref
#
if __name__ == "__main__":
    test_twobartruss_pdip()
    test_twobartruss_dual()
