## Imports
import numpy as np
import cvxpy as cp


class CVXPY:
    """
    This is a wrapper class to use the CVXOPT solver library found in the following link:
    https://cvxopt.org/userguide/solvers.html#problems-with-nonlinear-objectives.
    Requires an installation of ``cvxpy``.
    """
    def __init__(self, n, m):
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
                    \\alpha_i^{(k)} \\leq  x_i \\leq  \\beta_i^{(k)}  ,  i = 1, ..., n \\\\

                Input:  subprob
                Output: x
                """

        self.subprob = subprob

        # Design variables
        x = cp.Variable(self.n)

        # Objective function \tilde{g}_0
        cost = self.subprob.g(np.array(x).flatten())[0]
        obj = cp.Minimize(cost)

        # Constraint functions \tilde{g}_j <= 0, j=1,...,m


        # Linear inequality constraints (problem bounds)
        G = np.append(np.eye(self.n), -np.eye(self.n), axis=0)
        h = np.append(self.subprob.x_max, -self.subprob.x_min)
        bound_constraints = G*x <= h


        constr = []
        problem = cp.Problem(obj, constr)
        opt_value = problem.solve()
        solution = x.value






        return solvers.cp(self.F, G, h)['x']

    def F(self, x=None, z=None):
        """
        This function is required by the CVXOPT library, see https://cvxopt.org/userguide/solvers.html#s-cp.

        :param x:
        :param z:
        :return:
        """

        if x is None:
            x0 = matrix(0.5*(self.subprob.alpha+self.subprob.beta), (self.n, 1))
            return self.m, x0
        f = matrix(self.subprob.g(np.array(x).flatten()), (self.m+1, 1))
        Df = matrix(self.subprob.dg(np.array(x).flatten()), (self.m+1, self.n))
        if z is None:
            return f, Df
        DiagonalHessian = matrix(self.subprob.ddg(np.array(x).flatten()))
        H = spdiag(DiagonalHessian.T * z)
        return f, Df, H
