## Imports
import numpy as np

try:
    from cvxopt import solvers, matrix, spdiag
except ImportError:
    raise Exception(
        "Package cvxopt is not available in the current environment. "
        "Install cvxopt or install SAOR with feature flags cvxopt.")
    

class CVXOPT:
    """
    This is a wrapper class to use the CVXOPT solver library found in the following link:
    https://cvxopt.org/userguide/solvers.html#problems-with-nonlinear-objectives
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
                    \\x_min_i^{(k)} \\leq  x_i \\leq  \\x_max_i^{(k)}  ,  i = 1, ..., n \\\\

                Input:  subprob
                Output: x
                """

        self.subprob = subprob

        # Linear inequality constraints (problem bounds)
        G = matrix(np.append(np.eye(self.n), -np.eye(self.n), axis=0))
        h = matrix(np.append(self.subprob.x_max, -self.subprob.x_min), (2*self.n, 1))

        return solvers.cp(self.F, G, h)['x']

    def F(self, x=None, z=None):
        """
        This function is required by the CVXOPT library, see https://cvxopt.org/userguide/solvers.html#s-cp.

        :param x:
        :param z:
        :return:
        """

        if x is None:
            x0 = matrix(0.5*(self.subprob.x_min+self.subprob.x_max), (self.n, 1))
            return self.m, x0
        f = matrix(self.subprob.g(np.array(x).flatten()), (self.m+1, 1))
        Df = matrix(self.subprob.dg(np.array(x).flatten()), (self.m+1, self.n))
        if z is None:
            return f, Df
        DiagonalHessian = matrix(self.subprob.ddg(np.array(x).flatten()))
        H = spdiag(DiagonalHessian.T * z)
        return f, Df, H
