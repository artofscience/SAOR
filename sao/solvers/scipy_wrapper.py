## Imports
from scipy import optimize

"""
This is a wrapper class to use the SCIPY optimization library found in the following link:
https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize.
Requires an installation of ``scipy``.
"""


def scipy_solver(problem, **kwargs):
    """
        This function solves a given problem P:

        minimize

        .. math::

            \\tilde{g}_0^{(k)}[mathbf{x}]

        subject to

        .. math::

            \\tilde{g}_j^{(k)}[mathbf{x}] \\leq 0  ,  j = 1, ..., m \\\\
            \\x_min_i^{(k)} \\leq  x_i \\leq  \\x_max_i^{(k)}  ,  i = 1, ..., n \\\\

        Input:  problem
        Output: x
        """

    x0 = kwargs.get('x0', 0.5*(problem.x_min + problem.x_max))
    bounds = optimize.Bounds(problem.x_min, problem.x_max)
    objective = lambda x: problem.g(x)[0]
    objective_der = lambda x: problem.dg(x)[0]
    constraints = lambda x: -problem.g(x)[1:]
    constraints_der = lambda x: -problem.dg(x)[1:]
    # TODO: Possibly add diagonal Hessian that we currently have (they use a full Hessian with different dimensions)

    ineq_cons = {'type': 'ineq',
                 'fun': constraints,
                 'jac': constraints_der,
                 }
    method = kwargs.get('method', 'SLSQP')
    options = kwargs.get('options', None)
    solution = optimize.minimize(objective, x0, bounds=bounds, method=method,
                                 jac=objective_der, constraints=ineq_cons, options=options)
    return solution.x
