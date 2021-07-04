import pytest
import numpy as np
import logging
from Problems.Square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.intervening_variables import Linear, ConLin, Reciprocal, MMA
from sao.move_limits.move_limit import MoveLimit
from sao.problems.subproblem import Subproblem
from sao.solvers.interior_point import InteriorPointXYZ as ipxyz

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

np.set_printoptions(precision=4)


@pytest.mark.parametrize('n', [10])
def test_square_ipxyz(n):
    logger.info("Solving test_square using Ipopt xyz")

    # Instantiate problem
    prob = Square(n)
    assert prob.n == n

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(approximation=Taylor1(intervening=MMA(prob.xmin, prob.xmax)),
                         limits=MoveLimit(0.2, xmin=prob.xmin, xmax=prob.xmax))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Optimization loop
    while not (x_k == pytest.approx(1 / n * np.ones_like(x_k), rel=1e-3)):
        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Print current iteration and x_k
        logger.info('iter: {:^4d}  |  x: {:<20s}  |  obj: {:^9.3f}  |  constr: {:^6.3f}'.format(
            itte, np.array2string(x_k[0:2]), f[0], f[1]))

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df, ddf)
        solver = ipxyz(subprob, epsimin=1e-6)
        x_k = solver.update()

        itte += 1

    logger.info('Alles goed!')


if __name__ == "__main__":
    test_square_ipxyz(50)