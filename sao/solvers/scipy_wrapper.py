## Imports
from scipy import optimize


class SCIPY:
    """
    This is a wrapper class to use the SCIPY optimization library found in the following link:
    https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize.
    Requires an installation of ``scipy``.
    """
    def __init__(self, n, m, **kwargs):
        self.n = n
        self.m = m
        self.subprob = None

    def subsolv(self, subprob, **kwargs):
        """
                This method solves the approximate subproblem P_NLP_tilde:

                minimize

                .. math::

                    \\tilde{g}_0^{(k)}[mathbf{x}]

                subject to

                .. math::

                    \\tilde{g}_j^{(k)}[mathbf{x}] \\leq 0  ,  j = 1, ..., m \\\\
                    \\x_min_i^{(k)} \\leq  x_i \\leq  \\x_max_i^{(k)}  ,  i = 1, ..., n \\\\

                Input:  subprob
                Output: x
                """

        self.subprob = subprob
        x0 = kwargs.get('x0', 0.5*(self.subprob.x_min + self.subprob.x_max))
        bounds = optimize.Bounds(subprob.x_min, subprob.x_max)
        # ineq_cons = optimize.NonlinearConstraint(self.constraints, -np.inf, 0, jac=self.constraints_der)
        ineq_cons = {'type': 'ineq',
                     'fun' : self.constraints,
                     'jac' : self.constraints_der}
        solution = optimize.minimize(self.objective, x0, bounds=bounds, method='SLSQP',
                                     jac=self.objective_der, constraints=ineq_cons, options={'ftol': 1e-8})
        return solution.x

    def objective(self, x):
        return self.subprob.g(x)[0]

    def objective_der(self, x):
        return self.subprob.dg(x)[0]

    def constraints(self, x):
        """ The minus sign is there because inequality constraints must be in the form g_j >= 0."""
        return -self.subprob.g(x)[1:]

    def constraints_der(self, x):
        """ The minus sign is there because inequality constraints must be in the form g_j >= 0."""
        return -self.subprob.dg(x)[1:]

    # TODO: Possibly add diagonal Hessian that we currently have (they use a full Hessian with different dimensions)
