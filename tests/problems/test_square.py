import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.approximations.intervening import Linear, ConLin, MMA
from sao.move_limits.ml_intervening import MoveLimitIntervening
from sao.problems.subproblem import Subproblem
from sao.solvers.interior_point_basis import InteriorPointBasis as ipb
from sao.solvers.interior_point_artificial import InteriorPointArtificial as ipa
from sao.solvers.SolverIP_Svanberg import SvanbergIP

np.set_printoptions(precision=4)


@pytest.mark.parametrize('n', [10, 20])
def test_square(n):

    # Instantiate problem
    prob = Square(n)
    assert prob.n == n

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor2(),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))
    # subprob = Subproblem(intervening=ConLin(), approximation=Taylor2(),
    #                      ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))

    # Instantiate solver
    solver = SvanbergIP(prob.n, 1)

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x.copy()

    # Optimization loop
    while not (x_k == pytest.approx(1/n * np.ones_like(x_k), rel=1e-4)):

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Print current iteration and x_k
        print('iter: {:^4d}  |  x: {:<20s}  |  obj: {:^9.3f}  |  constr: {:^6.3f}'.format(itte, np.array2string(x_k[0:2]), f[0], f[1]))

        # Build approximate sub-problem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Call solver (x_k, g and dg are within approx instance)
        x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)
        x_k = x.copy()

        # solver = ipa(subprob, epsimin=1e-7)
        # solver.update()
        # x_k = solver.x.copy()

        itte += 1

    print('Alles goed!')


if __name__ == "__main__":
    test_square(4)

