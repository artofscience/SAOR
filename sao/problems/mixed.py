"""Mixed scheme subproblem implementation."""
import numpy as np
from sao.problems.subproblem import Subproblem


class Mixed(Subproblem):
    """Implements a subproblem with multiple variable and response sets.

    TODO: improve documentation on input types and the formulation of the
    required subproblem, variable, and response mappings.
    """

    def __init__(self, subproblems, variables, responses):
        """Store response/variable mapping and initialise arrays."""
        self.subproblems = subproblems
        self.variables = variables
        self.responses = responses

        self.n = sum(len(v) for v in self.variables.values())
        self.m = sum(len(r) for r in self.responses.values())

        self.alpha = np.zeros(self.n, dtype=float)
        self.beta = np.ones(self.n, dtype=float)
        self.f = np.zeros(self.m, dtype=float)
        self.df = np.zeros((self.m, self.n), dtype=float)
        self.ddf = np.zeros((self.m, self.n), dtype=float)

    def __getitem__(self, key):
        """Get subproblem by indexing on class instance."""
        return self.subproblems[key]

    def __setitem__(self, key, value):
        """Set subproblem value by indexing on class instance."""
        self.subproblems[key] = value

    def items(self):
        """Yields all indices and values of the responses and variable sets.

        This is a shorthand for iterating of the product of the (keys, values)
        of both the variable and response sets. It mostly acts to reduce the
        double loops throughout this class.
        """
        for (i, response) in self.responses.items():
            for (j, variable) in self.variables.items():
                yield (i, j), (response, variable)

    def build(self, x, f, df, ddf=None):
        """Builds the subproblem for all variable and response sets."""
        self.n, self.m = len(x), len(f) - 1
        self.f, self.df, self.ddf = f, df, ddf

        # The mixed subproblem is constructed by building each of the defined
        # subproblems within this mixed instance. This loop runs over all
        # available response `i, r` and variable `j, v` sets and invokes the
        # corresponding subproblems obtained by the combined keys `i, j` into
        # the subproblem index. The second order derivatives `ddf` need only
        # be considered when they are set.
        for ((i, j), (r, v)) in self.items():
            ddf = None if ddf is None else self.ddf[np.ix_(r, v)]
            self[i, j].build(x[v], f[r], df[np.ix_(r, v)], ddf)

        # The alpha/beta bounds are build by merging the alpha/beta bounds for
        # each of the defined subproblems in the mixed subproblem
        self.alpha[:] = -np.inf
        self.beta[:] = np.inf

        for ((i, j), (r, v)) in self.items():
            self.alpha[v] = np.maximum(self.alpha[v], self[i, j].alpha)
            self.beta[v] = np.minimum(self.beta[v], self[i, j].beta)

    def g(self, x):
        """Approximate the function value across all responses."""
        # Each time a variables is added: so should the constant term
        result = - (len(self.variables) - 1) * self.f

        for ((i, j), (r, v)) in self.items():
            result[r] += self[i, j].approx.g(self[i, j].inter.y(x[v]).T)

        return result

    def dg(self, x):
        """Approximate the derivative across all responses/variables."""
        result = np.zeros_like(self.df)

        for ((i, j), (r, v)) in self.items():
            result[np.ix_(r, v)] = self[i, j].approx.dg(
                self[i, j].inter.y(x[v]).T,
                self[i, j].inter.dydx(x[v]))

        return result

    def ddg(self, x):
        """Approximate the second derivative of all responses/variables."""
        result = np.zeros_like(self.df)
        for ((i, j), (r, v)) in self.items():
            result[np.ix_(r, v)] = self[i, j].approx.ddg(
                self[i, j].inter.y(x[v]).T,
                self[i, j].inter.dydx(x[v]),
                self[i, j].inter.ddyddx(x[v]))

        return result
