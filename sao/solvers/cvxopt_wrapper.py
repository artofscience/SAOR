## Imports
import numpy as np
from cvxopt import solvers, matrix, spdiag


class CVXOPTwrapper:
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
        This function subsolv solves the approximate subproblem P_NLP_tilde:

        minimize

        .. math::

            \\tilde{g}_0^{(k)}[mathbf{x}]

        subject to

        .. math::

            \\tilde{g}_j^{(k)}[mathbf{x}] \\leq 0  ,  j = 1, ..., m \\\\
            \\alpha_i^{(k)} \\leq  x_i \\leq  \\beta_i^{(k)}  ,  i = 1, ..., n \\\\

        Input:  subprob
        Output: x           # TODO: what will the output be?
        """

        self.subprob = subprob
        return solvers.cp(self.F(x=0.5 * (subprob.alpha + subprob.beta)))['x']

    def F(self, x=None):
        if x is None: return 0, matrix(0.0, (self.n, 1))
        f = self.subprob.g(x)
        Df = self.subprob.dg(x)
        return f, Df
