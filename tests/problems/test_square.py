import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.taylor import Taylor1
from sao.approximations.intervening import MMA
from sao.move_limits.move_limit import MoveLimitStrategy
from sao.subproblems.subproblem import Subproblem
from sao.solvers.interior_point_basis import InteriorPointBasis as ipa

np.set_printoptions(precision=4)


@pytest.mark.parametrize('n', [2, 5, 10])
def test_square(n):

    # Instantiate problem
    prob = Square(n)
    assert prob.n == n

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor1(),
                        ml=MoveLimitStrategy(xmin=prob.xmin, xmax=prob.xmax))
    subprob.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x.copy()

    # Optimization loop
    while not (x_k == pytest.approx(1/n * np.ones_like(x_k), rel=1e-4)):

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = prob.ddg(x_k)

        # Print current iteration and x_k
        print('iter: {:<4d}  |  obj: {:>9.3f}  |  constr: {:>6.3f}'.format(itte, f[0], f[1]))

        # Build approximate sub-problem at X^(k)
        subprob.update_approximation(prob.x, f, df, ddf)

        # Call solver (x_k, g and dg are within approx instance)
        # x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(approx)
        solver = ipa(subprob, epsimin=1e-9)
        solver.update()

        x_k = solver.x.copy()
        itte += 1

    print('Alles goed!')


if __name__ == "__main__":
    test_square(4)

