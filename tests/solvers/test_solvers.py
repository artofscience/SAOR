import pytest
import numpy as np
import logging
from Problems.square import Square
from sao.solvers.interior_point_xyz import InteriorPointArtificial as ipa
from sao.solvers.interior_point_x import InteriorPointBasis as ipb
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

    # Test sao.solvers.interior_point_x.py
    logger.info("Solve x**2 using ipopt basis")
    problemb = Square(n)
    mysolverb = ipb(problemb, epsimin=1e-7)
    mysolverb.update()
    assert np.sum(mysolverb.x) == pytest.approx(1, rel=1e-4)

    # Test sao.solvers.interior_point_xyz.py
    logger.info("Solve x**2 using ipopt with artificial variables")
    problema = Square(n)
    mysolvera = ipa(problema, epsimin=1e-9)
    mysolvera.update()
    assert np.sum(mysolvera.x) == pytest.approx(1, rel=1e-4)

    # Test sao.solvers.SolverIP_Svanberg.py
    logger.info("Solve x**2 using SvanbergIP")
    problemc = Square(n)
    mysolverc = SvanbergIP(problemc.n, 1)
    x, y, z, lam, xsi, eta, mu, zet, s = mysolverc.subsolv(problemc)
    assert np.sum(x) == pytest.approx(1, rel=1e-4)

    # Compare results of solvers
    assert np.linalg.norm(mysolverb.x - mysolvera.x) == pytest.approx(0, abs=1e-4)
    assert np.linalg.norm(x - mysolvera.x) == pytest.approx(0, abs=1e-4)


if __name__ == "__main__":
    test_square(10)
