from sao.move_limits import GeneralMoveLimit, Bound
from sao.util.tools import fill_set_when_emtpy
import numpy as np


class Mixed(GeneralMoveLimit):
    """
    For variable, a separate or combination of
    move limits can be set. (movelimit, variable).

    The variables are tracked by a dictionary of move limit indices to variable
    sets. So, ``{0: {0, 1, 3}, 1: {0, 2}}`` indicates that for move limit ``0``
    the variables ``{0, 1, 3}`` are relevant and for move limit ``1`` only the
    variable set ``{0, 2}``. The variable sets used in different move limit strategies can
    overlap.
    """

    def __init__(self, nvar: int, default: GeneralMoveLimit = Bound()):
        self.default = default
        self.nvar = nvar
        self.ml_mapping = [tuple]

        # On initialisation the default move limit is added to all variables
        variables = set(range(self.nvar))
        self.add_move_limit(self.default, variables)

    @property
    def move_limits(self):
        """Yields the move limits."""
        for ml, _ in self.ml_mapping:
            yield ml

    def set_move_limit(self, ml: GeneralMoveLimit, var=Ellipsis):
        """Assign a move limit strategy to some variables.

        Other move limits that might be pointing to the same
        variables are updated accordingly to avoid any overlap between the
        different response sets.
        """

        new_vars = fill_set_when_emtpy(var, self.nvar)

        for _, variables in self.ml_mapping:
            # Only consider to remove entries when the new response shares
            # the same indices as the existing responses (set intersection).
            if len(diff := variables - new_vars) > 0:
                # If the resulting set of variables is non-empty, we need
                # to add the the variables to the current set with the
                # remaining variables.
                variables = diff
            else:
                # If the resulting set is empty, the
                # corresponding variables can be deleted from the mapping.
                del variables

        # After deleting the overlapping regions in any other variable sets
        # an additional move limit is added.
        return self.add_move_limit(ml, new_vars)

    def add_move_limit(self, ml: GeneralMoveLimit, var=Ellipsis):
        variables = fill_set_when_emtpy(var, self.nvar)
        self.ml_mapping.append((ml, variables))
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
        for ml, variables in self.ml_mapping:
            x[list(variables)] = ml.clip(x[list(variables)])
        return x
