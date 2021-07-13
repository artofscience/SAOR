from sao.move_limits import GeneralMoveLimit, Bound
import numpy as np


class Mixed(GeneralMoveLimit):
    """
    For every variable j, a separate or combination of move limits can be set.
    (move limit, variable)
    """

    def __init__(self, nvar: int, default: GeneralMoveLimit = Bound()):
        self.default = default
        self.nvar = nvar
        self.all_ml = []
        self.all_var = np.arange(self.nvar, dtype=int)
        if default is not None:
            self.all_ml = [(self.default, [self.all_var])]

    @property
    def move_limits(self):
        """Yields the move limits."""
        for ml, _ in self.all_ml:
            yield ml

    def set_move_limit(self, ml: GeneralMoveLimit, var=Ellipsis):
        the_vars = np.unique(np.atleast_1d(self.all_var[var]))
        # Remove from existing move limits
        for dat in self.all_ml:
            dat[1][:] = np.setdiff1d(dat[1][:], the_vars, assume_unique=True)
        self.all_ml.append((ml, [the_vars]))
        return self

    def add_move_limit(self, ml: GeneralMoveLimit, var=Ellipsis):
        the_vars = np.unique(np.atleast_1d(self.all_var[var]))
        self.all_ml.append((ml, [the_vars]))
        return self

    def update(self, *args, **kwargs):
        """Perform inplace updates of the state of the move limits.

        This allows to perform additional functionality to update the state
        of the move limits, for instance to keep track of information
        at previous iterations etc.
        """
        for ml in self.move_limits:
            ml.update(*args, **kwargs)
        return self

    def clip(self, x):
        """Clips ``x`` with bounds of each move limit."""
        for ml in self.move_limits:
            ml.clip(x)
        return x
