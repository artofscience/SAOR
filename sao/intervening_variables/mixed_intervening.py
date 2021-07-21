from sao.intervening_variables import Intervening, Linear
import numpy as np


def fill_set_when_emtpy(s, n):
    """Returns ``set(s)`` or a ``set(0..n)`` if ``set(s)`` is the empty set."""
    if s is None or s is ...:
        return set(range(n))

    try:
        s = set(s)
    except TypeError:
        s = set([s])

    if len(s) == 0:
        return set(range(n))
    return s


class MixedIntervening(Intervening):
    """
    For every response i, and variable j, a separate or combination of
    intervening variables can be set. (intervening, response, variable).

    The responses are tracked by sets of indices that indicate the response of
    interest for the given intervening variable. These sets are
    non-overlapping, i.e. there is only one intervening variable that points to
    a response at all times.

    The variables are tracked by a dictionary of response indices to variable
    sets. So, ``{0: {0, 1, 3}, 1: {0, 2}}`` indicates that for response ``0``
    the variables ``{0, 1, 3}`` are relevant and for response ``1`` only the
    variable set ``{0, 2}``. The variable sets used in different responses can
    overlap.
    """

    def __init__(self, nvar: int, nresp: int, default: Intervening = Linear()):
        self.default = default
        self.nvar = nvar
        self.nresp = nresp
        self.iv_mapping = []

        # On initialisation the default intervening variable is added to all
        # the responses pointing to all considered variables.
        responses = set(range(self.nresp))
        variables = set(range(self.nvar))
        self.add_intervening(self.default, responses, variables)

    @property
    def intervening_variables(self):
        """Yields only the intervening variables."""
        for intv, _, _ in self.iv_mapping:
            yield intv

    def set_intervening(self, inter: Intervening, resp=set(), var=set()):
        """Assign a intervening variable to some variables/responses.

        Other intervening variables that might be pointing to the same
        responses are updated accordingly to avoid any overlap between the
        different response sets.
        """
        new_resp = fill_set_when_emtpy(resp, self.nresp)
        new_vars = fill_set_when_emtpy(var, self.nvar)

        for _, responses, variables in self.iv_mapping:
            # Only consider to remove entries when the new response shares
            # the same indices as the existing responses (set intersection).
            for r in (new_resp & responses):
                diff = len(variables[r] - new_vars)
                if diff > 0:
                    # If the resulting set of variables is non-empty, we need
                    # to add the index `r` to the current set with the
                    # remaining variables.
                    responses.add(r)
                    variables[r] = diff
                else:
                    # If the resulting set is empty, the index `r` can be
                    # removed from the current set of responses and the
                    # corresponding variables can be deleted from the mapping.
                    responses.remove(r)
                    del variables[r]

        # After deleting the overlapping regions in any other response and/or
        # variable sets, an additional intervening variable is added.
        return self.add_intervening(inter, new_resp, new_vars)

    def add_intervening(self, inter, resp=set(), var=set()):
        """Adds an additional intervening variable to responses and variables.

        The mapping only considers the unique set of elements in the response
        and variable sets. When an empty is given, all responses/variables will
        be considered.
        """
        responses = fill_set_when_emtpy(resp, self.nresp)
        variables = fill_set_when_emtpy(var, self.nvar)

        self.iv_mapping.append(
            (inter, responses, {i: variables for i in responses})
        )
        return self

    def evaluate_for_each_response(self, x, fn: callable):
        """Evaluates a function for each response and collects its output.

        Allocates the output of size ``number of reponses`` by ``number of
        design variables`` and populates the output by evaluating a callable
        function for each intervening variable given the current ``x``.
        """
        out = np.zeros((self.nresp, x.shape[0]))
        for intv, responses, variables in self.iv_mapping:
            y_all = fn(intv, x)
            for r in responses:
                var_indices = list(variables[r])
                if y_all.ndim > 1:
                    out[r, var_indices] += y_all[r, var_indices]
                else:
                    out[r, var_indices] += y_all[var_indices]
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
        for intv in self.intervening_variables:
            intv.update(*args, **kwargs)
        return self

    def clip(self, x):
        """Clips ``x`` with bounds of each intervening variable."""
        for intv in self.intervening_variables:
            intv.clip(x)
        return x
