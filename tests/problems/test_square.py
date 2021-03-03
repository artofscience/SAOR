import pytest
import numpy as np
from Problems.square import Square
from sao.approximations.taylor import Taylor1, Taylor2
from sao.approximations.intervening import Linear, Reciprocal, ConLin, MMA
from sao.approximations.bounds import Bounds
from sao.approximations.interveningapproximation import InterveningApproximation
from sao.solvers.SolverIP_Svanberg import SvanbergIP
from sao.solvers.interior_point_basis import InteriorPointBasis as ipb
from sao.solvers.interior_point_basis import InteriorPointBasis as ipa

np.set_printoptions(precision=4)


@pytest.mark.parametrize('n', [2, 5, 10])
def test_square(n):

    # Instantiate problem
    prob = Square(n)
    assert prob.n == n

    # Instantiate a non-mixed approximation scheme
    approx = InterveningApproximation(intervening=MMA(prob.xmin, prob.xmax), approximation=Taylor2(),
                                      bounds=Bounds(prob.xmin, prob.xmax))
    approx.update_approximation(prob.x, prob.g(prob.x), prob.dg(prob.x), prob.ddg(prob.x))

    # Instantiate solver
    # solver = SvanbergIP(n, 1)
    solver = ipa(approx, epsimin=1e-9)
    solver.update()

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x.copy()

    # Optimization loop
    while not x_k == pytest.approx(1/n * np.ones_like(x_k), rel=1e-4):

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k))
        g = prob.g(x_k)
        dg = prob.dg(x_k)
        ddg = prob.ddg(x_k)

        # Print current iteration and x_k
        print('iter: {:<4d}  |  obj: {:>9.3f}  |  constr: {:>6.3f}'.format(itte, g[0], g[1]))

        # Build approximate sub-problem at X^(k)
        approx.update_approximation(prob.x, prob.g(prob.x), dg, ddg)

        # Call solver (x_k, g and dg are within approx instance)
        # x, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(approx)
        solver = ipa(approx, epsimin=1e-9)
        solver.update()

        x_k = solver.x
        itte += 1

    print('Alles goed!')


if __name__ == "__main__":
    test_square(4)

