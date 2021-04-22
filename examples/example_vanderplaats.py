import pytest
import numpy as np
import logging
from Problems.VanderplaatsBeam import Vanderplaats
from sao.approximations.taylor import Taylor1, Taylor2
from sao.intervening_vars.intervening import Linear, ConLin, MMA, MMASquared
from sao.move_limits.move_limit import MoveLimitIntervening
from sao.problems.subproblem import Subproblem
from sao.solvers.interior_point import InteriorPointX as ip_x
from sao.solvers.interior_point import InteriorPointXY as ip_xy
from sao.solvers.interior_point import InteriorPointXYZ as ip_xyz
from sao.solvers.SolverIP_Svanberg import SvanbergIP
from sao.solvers.interior_point import InteriorPointXYZ as ipopt

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

np.set_printoptions(precision=4)


@pytest.mark.parametrize('N', [50])
def test_vanderplaats(N):

    # Instantiate problem
    prob = Vanderplaats(N)
    assert prob.n == 2 * N

    # Instantiate a non-mixed approximation scheme
    # subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
    #                      ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    subprob = Subproblem(intervening=MMASquared(prob.xmin, prob.xmax), approximation=Taylor1(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))


    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()
    xold1 = np.zeros_like(x_k)
    vis = None
    solves = 0

    # Optimization loop
    while np.linalg.norm(x_k - xold1) > 1e-3:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)

        # Print current iteration and x_k
        vis = prob.visualize(x_k, itte, vis)
        logger.info(
            'iter: {:^4d}  |  obj: {:^9.3f}  |  constr1: {:^6.3f}  |  constr2: {:^6.3f}  |  constr3: {:^6.3f}'.format(
                itte, f[0], f[1], f[2], f[3]))

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df)

        # Call solver (x_k, g and dg are within approx instance)
        # x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)
        solver = ipopt(subprob, epsimin=1e-9, x0=x_k)
        x = solver.update()
        solves += solver.itera
        xold1 = x_k.copy()
        x_k = x.copy()


        itte += 1

    logger.info('Optimization loop converged!')
    print(solves)


if __name__ == "__main__":
    test_vanderplaats(100)
