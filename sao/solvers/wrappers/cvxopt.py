import numpy as np

try:
    from cvxopt import solvers, matrix, spdiag

    def cvxopt_solver(problem, **kwargs):
        """
        This is a wrapper function that uses the ``cvxopt`` solver library found in the following link:
        https://cvxopt.org/userguide/solvers.html#problems-with-nonlinear-objectives

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

        def F(x=None, z=None):
            """
            This function is required by the ``cvxopt`` library, see https://cvxopt.org/userguide/solvers.html#s-cp.

            :param x:
            :param z:
            :return:
            """

            if x is None:
                x0 = matrix(0.5 * (problem.x_min + problem.x_max), (problem.n, 1))
                return problem.m, x0
            f = matrix(problem.g(np.array(x).flatten()), (problem.m + 1, 1))
            Df = matrix(problem.dg(np.array(x).flatten()), (problem.m + 1, problem.n))
            if z is None:
                return f, Df
            DiagonalHessian = matrix(problem.ddg(np.array(x).flatten()))
            H = spdiag(DiagonalHessian.T * z)
            return f, Df, H

        # Linear inequality constraints (problem bounds)
        G = matrix(np.append(np.eye(problem.n), -np.eye(problem.n), axis=0))
        h = matrix(np.append(problem.x_max, -problem.x_min), (2 * problem.n, 1))

        return solvers.cp(F, G, h)['x']
except ImportError:
    def cvxopt_solver(problem, **kwargs):
        raise Exception(
            "Package cvxopt is not available in the current environment. "
            "Install cvxopt or install SAOR with feature flags cvxopt.")
