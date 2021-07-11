from sao.intervening_variables import Intervening, Linear
import numpy as np


class Mixed(Intervening):
    """
    For every response i, and variable j, a separate or combination of intervening variables can be set.
    (intervening, response, variable)
    """

    def __init__(self, nvar: int, nresp: int, default: Intervening = Linear()):
        self.default = default
        self.nvar = nvar
        self.nresp = nresp
        self.all_inter = []
        self.all_resp = np.arange(self.nresp, dtype=int)
        self.all_var = np.arange(self.nvar, dtype=int)
        if default is not None:
            self.all_inter = [(self.default, [True] * self.nresp, [self.all_var] * self.nresp)]

    def set_intervening(self, inter: Intervening, var=Ellipsis, resp=Ellipsis):
        which_var = [np.array([], dtype=int)] * self.nresp
        which_resp = [False] * self.nresp

        the_vars = np.unique(np.atleast_1d(self.all_var[var]))
        for i in np.atleast_1d(self.all_resp[resp]):
            which_var[i] = the_vars
            which_resp[i] = True
            # Remove from existing intervening variables
            for dat in self.all_inter:
                if dat[1][i]:
                    dat[2][i] = np.setdiff1d(dat[2][i], the_vars, assume_unique=True)
                    dat[1][i] = len(dat[2][i]) > 0

        self.all_inter.append((inter, which_resp, which_var))
        return self

    def add_intervening(self, inter: Intervening, var=Ellipsis, resp=Ellipsis):
        which_var = [np.array([], dtype=int)] * self.nresp
        which_resp = [False] * self.nresp

        the_vars = np.unique(np.atleast_1d(self.all_var[var]))
        for i in np.atleast_1d(self.all_resp[resp]):
            which_var[i] = the_vars
            which_resp[i] = True
        self.all_inter.append((inter, which_resp, which_var))
        return self

    def evaluate_for_each_response(self, x, fn: callable):
        """Evaluates a function for each response and collects its output.

        Allocates the output of size ``number of reponses`` by ``number of
        design variables`` and populates the output by evaluating a callable
        function for each intervening variable given the current ``x``.
        """
        out = np.zeros((self.nresp, x.shape[0]))
        for intv, which_resp, which_var in self.all_inter:
            y_all = fn(intv, x)
            for r in self.all_resp[which_resp]:
                if y_all.ndim > 1:
                    out[r, which_var[r]] += y_all[r, which_var[r]]
                else:
                    out[r, which_var[r]] += y_all[which_var[r]]
        return out

    def y(self, x):
        """Evaluates the mapping y = f(x)."""
        def y_of_x(cls, x):
            return cls.y(x)
        return self.evaluate_for_each_response(x, y_of_x)

    def dydx(self, x):
        """Evaluates the first derivative of the mapping at x."""
        def dy_of_x(cls, x):
            return cls.dydx(x)
        return self.evaluate_for_each_response(x, dy_of_x)

    def ddyddx(self, x):
        """Evaluates the second derivatives of the mapping at x."""
        def ddy_of_x(cls, x):
            return cls.ddyddx(x)
        return self.evaluate_for_each_response(x, ddy_of_x)

    def update(self, *args, **kwargs):
        """Perform inplace updates of the state of the intervening variable.

        This allows to perform additional functionality to update the state
        of the intervening variable, for instance to keep track of information
        at previous iterations etc.
        """
        for intv, _, _ in self.all_inter:
            intv.update(*args, **kwargs)
        return self

    def clip(self, x):
        """Make constituent clips."""
        for intv, _, _ in self.all_inter:
            intv.clip(x)
        return x
