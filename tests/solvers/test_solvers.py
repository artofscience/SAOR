import pytest
import numpy as np
import logging
from Problems.nd.Square import Square
from sao.solvers.primal_dual_interior_point import pdip, Pdipx, Pdipxy, Pdipxyz
from sao.solvers.SolverIP_Svanberg import SvanbergIP

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@pytest.mark.parametrize('n', [100])
def test_square(n):
    # Test sao.solvers.interior_point.py
    logger.info("Solve x**2 using ipopt with x, y, z")
    problem_xyz = Square(n)
    mysolver_xyz = pdip(problem_xyz, epsimin=1e-9)
    assert np.sum(mysolver_xyz) == pytest.approx(1, rel=1e-4)

    logger.info("Solve x**2 using ipopt with x, y")
    problem_xy = Square(n)
    mysolver_xy = pdip(problem_xy, variables=Pdipxy, epsimin=1e-9)
    assert np.sum(mysolver_xy) == pytest.approx(1, rel=1e-4)

    logger.info("Solve x**2 using ipopt with x")
    problem_x = Square(n)
    mysolver_x = pdip(problem_x, variables=Pdipx, epsimin=1e-9)
    assert np.sum(mysolver_x) == pytest.approx(1, rel=1e-4)

    # Test sao.solvers.SolverIP_Svanberg.py
    logger.info("Solve x**2 using SvanbergIP")
    problem_svan = Square(n)
    mysolver_svan = SvanbergIP(problem_svan.n, 1)
    x, y, z, lam, xsi, eta, mu, zet, s = mysolver_svan.subsolv(problem_svan)
    assert np.sum(x) == pytest.approx(1, rel=1e-4)

    # Compare results of solvers
    assert np.linalg.norm(mysolver_xyz - mysolver_xy) == pytest.approx(0, abs=1e-4)
    assert np.linalg.norm(x - mysolver_x) == pytest.approx(0, abs=1e-4)


if __name__ == "__main__":
    test_square(10)
