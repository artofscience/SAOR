import logging

import numpy as np
import pytest

from problems.n_dim.square import Square
from sao.solvers.pdip_svanberg import ipsolver
from sao.solvers.wrappers.cvxopt import cvxopt_solver
from sao.solvers.primal_dual_interior_point import pdip, Pdipx, Pdipxy, Pdipxyz
from sao.solvers.wrappers.scipy import scipy_solver

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
    x_opt_xyz = pdip(problem_xyz, variables=Pdipxyz, epsimin=1e-7)[0]
    assert np.sum(x_opt_xyz) == pytest.approx(1, rel=1e-4)

    logger.info("Solve x**2 using ipopt with x, y")
    problem_xy = Square(n)
    x_opt_xy = pdip(problem_xy, variables=Pdipxy, epsimin=1e-7)[0]
    assert np.sum(x_opt_xy) == pytest.approx(1, rel=1e-4)

    logger.info("Solve x**2 using ipopt with x")
    problem_x = Square(n)
    x_opt_x = pdip(problem_x, variables=Pdipx, epsimin=1e-7)[0]
    assert np.sum(x_opt_x) == pytest.approx(1, rel=1e-4)

    # Test sao.solvers.pdip_svanberg.py
    logger.info("Solve x**2 using SvanbergIP")
    problem_svan = Square(n)
    x_opt_svan = ipsolver(problem_svan, epsimin=1e-7)
    assert np.sum(x_opt_svan) == pytest.approx(1, rel=1e-4)

    # Test sao.solvers.cvxopt.py
    logger.info("Solve x**2 using cvxopt")
    problem_cvxopt = Square(n)
    x_opt_cvxopt = cvxopt_solver(problem_cvxopt)
    assert np.sum(x_opt_cvxopt) == pytest.approx(1, rel=1e-4)

    # Test sao.solvers.scipy.py
    logger.info("Solve x**2 using scipy")
    problem_scipy = Square(n)
    x_opt_scipy = scipy_solver(problem_scipy)
    assert np.sum(x_opt_scipy) == pytest.approx(1, rel=1e-4)

    # Compare results of solvers
    assert np.linalg.norm(x_opt_xyz - x_opt_xy) == pytest.approx(0, abs=1e-4)
    assert np.linalg.norm(x_opt_svan - x_opt_x) == pytest.approx(0, abs=1e-4)
    assert np.linalg.norm(x_opt_cvxopt - x_opt_svan) == pytest.approx(0, abs=1e-4)
    assert np.linalg.norm(x_opt_scipy - x_opt_cvxopt) == pytest.approx(0, abs=1e-4)


if __name__ == "__main__":
    test_square(10)
