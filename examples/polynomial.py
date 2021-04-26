import numpy as np
import logging
from Problems.QuadPoly1D import QuadPoly1D
from sao.approximations.taylor import Taylor1, Taylor2, SphericalTaylor2, NonSphericalTaylor2
from sao.intervening_vars.intervening import Linear, ConLin, MMA
from sao.move_limits.move_limit import MoveLimitIntervening
from sao.problems.subproblem import Subproblem
from sao.solvers.SolverIP_Svanberg import SvanbergIP
from sao.util.plotter import Plot, Plot2

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

np.set_printoptions(precision=4)


def example_poly():
    logger.info("Solving test_poly using Ipopt Svanberg")

    # Instantiate problem
    prob = QuadPoly1D()

    # Instantiate a non-mixed approximation scheme
    subprob = Subproblem(intervening=Linear(), approximation=NonSphericalTaylor2(force_convex=False),
                         ml=MoveLimitIntervening(xmin=prob.xmin, xmax=prob.xmax))

    # Initialize iteration counter and design
    itte = 0
    x_k = prob.x0.copy()

    # Instantiate solver
    solver = SvanbergIP(prob.n, prob.m)

    # Instantiate plotter
    plotter = Plot(['objective', 'constraint_1'], path=".")
    plotter2_flag = True
    if plotter2_flag:
        plotter2 = Plot2(prob)

    # Optimization loop
    while itte < 50:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = prob.g(x_k)
        df = prob.dg(x_k)
        ddf = (prob.ddg(x_k) if subprob.approx.__class__.__name__ == 'Taylor2' else None)

        # Print & plot g_j and x_i at current iteration
        logger.info('iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}'.format(itte, np.array2string(x_k[0]), f[0]))
        plotter.plot([f[0], f[1]])

        # Build approximate subproblem at X^(k)
        subprob.build(x_k, f, df, ddf)

        # Plot current approximation
        if plotter2_flag:
            plotter2.plot_approx(x_k, f, prob, subprob, itte)

        # Solve current subproblem
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subprob)

        itte += 1

    logger.info('Alles goed!')


if __name__ == "__main__":
    example_poly()
