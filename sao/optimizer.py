import numpy as np
import logging
import sao

# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
np.set_printoptions(precision=4)


def optimize(problem, solver, approximation, converged, plotter=None, *args, **kwargs):
    """
    This is a wrapper function for the main file of an optimization.
    Takes as arguments the following objects and performs the optimization main loop.

    :param problem: An object that holds the initial problem to be solved.
    :param solver: An object that holds the solver to be used.
    :param approximation: An object that holds the approximation (and the intervening variables) to be used.
    :param converged: An object that holds the convergence criterion.
    :param plotter: An object that performs plotting functionalities as the optimization runs.
    :param args:
    :param kwargs:
    :return:
    """

    logger.info(f"Solving {problem.__class__.__name__} using "
                f"{approximation.interv[0].__class__.__name__} intervening variables "
                f"and {solver.__name__} solver")

    # Instantiate the subproblem       # TODO: improve imports (didn't want to use import *)
    subproblem = sao.problems.subproblem.Subproblem(approximation=approximation)
    subproblem.set_limits([sao.move_limits.Bounds(problem.xmin, problem.xmax),
                           sao.move_limits.MoveLimit(move_limit=0.1, dx=problem.xmax - problem.xmin)])

    # Initialize design and iteration counter
    x_k = kwargs.get('x0', problem.x0)
    itte = 0

    # Optimization loop
    while not converged:

        # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
        f = problem.g(x_k)
        df = problem.dg(x_k)
        ddf = problem.ddg(x_k) if isinstance(approximation, sao.approximations.Taylor2) else None

        # Build approximate sub-problem at X^(k)
        subproblem.build(x_k, f, df, ddf)

        # Call solver (x_k, g and dg are within approx instance)
        x_k, y, z, lam, xsi, eta, mu, zet, s = solver.subsolv(subproblem)
        x_k,  = solver.subsolv(subproblem)

        # Print & Plot              # TODO: Print and Plot the criterion as criterion.value (where 0 is now)
        logger.info(
            'iter: {:^4d}  |  x: {:<10s}  |  obj: {:^9.3f}  |  criterion: {:^6.3f}  |  max_constr_viol: {:^6.3f}'.format(
                itte, np.array2string(x_k[0]), f[0], 0, max(0, max(f[1:]))))
        if plotter is not None:
            plotter.plot([f[0], f[1], 0, max(0, max(f[1:]))])

        itte += 1

    logger.info('Optimization loop converged!')