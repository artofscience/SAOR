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
            self.all_inter.append( (default, [True for _ in self.all_resp], [self.all_var for _ in range(self.nresp)]) )

    def set_intervening(self, inter: Intervening, var=Ellipsis, resp=Ellipsis):
        which_var = [np.array([], dtype=int) for _ in range(self.nresp)]
        which_resp = [False for _ in range(self.nresp)]

        the_vars = np.unique(np.atleast_1d(self.all_var[var]))
        for i in np.atleast_1d(self.all_resp[resp]):
            which_var[i] = the_vars
            which_resp[i] = True
            # Remove from existing intervening variables
            for dat in self.all_inter:
                if not dat[1][i]:
                    continue
                dat[2][i] = np.setdiff1d(dat[2][i], the_vars, assume_unique=True)
                dat[1][i] = len(dat[2][i]) > 0

        self.all_inter.append((inter, which_resp, which_var))
        return self

    def add_intervening(self, inter: Intervening, var=Ellipsis, resp=Ellipsis):
        which_var = [np.array([], dtype=int) for _ in range(self.nresp)]
        which_resp = [False for _ in range(self.nresp)]

        the_vars = np.unique(np.atleast_1d(self.all_var[var]))
        for i in np.atleast_1d(self.all_resp[resp]):
            which_var[i] = the_vars
            which_resp[i] = True
        self.all_inter.append((inter, which_resp, which_var))
        return self

    def y(self, x):
        """Evaluates the mapping y = f(x)."""
        assert x.ndim == 1, "Only for 1-dimensional x"
        out = np.zeros((self.nresp, x.shape[0]))
        for intv, which_resp, which_var in self.all_inter:
            y_all = intv.y(x)
            for i, r in enumerate(self.all_resp[which_resp]):
                if y_all.ndim > 1:
                    out[r, which_var[r]] += y_all[i, which_var[r]]
                else:
                    out[r, which_var[r]] += y_all[which_var[r]]
        return out

    def dydx(self, x):
        """Evaluates the first derivative of the mapping at x."""
        out = np.zeros((self.nresp, x.shape[0]))
        for intv, which_resp, which_var in self.all_inter:
            dy_all = intv.dydx(x)
            for i, r in enumerate(self.all_resp[which_resp]):
                if dy_all.ndim > 1:
                    out[r, which_var[r]] += dy_all[i, which_var[r]]
                else:
                    out[r, which_var[r]] += dy_all[which_var[r]]
        return out

    def ddyddx(self, x):
        """Evaluates the second derivatives of the mapping at x."""
        out = np.zeros((self.nresp, x.shape[0]))
        for intv, which_resp, which_var in self.all_inter:
            ddy_all = intv.ddyddx(x)
            for i, r in enumerate(self.all_resp[which_resp]):
                if ddy_all.ndim > 1:
                    out[r, which_var[r]] += ddy_all[i, which_var[r]]
                else:
                    out[r, which_var[r]] += ddy_all[which_var[r]]
        return out

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
        """Default clipping is none."""
        return x
