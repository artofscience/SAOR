def parse_to_list(*args):
    if len(args) == 0:
        return []
    elif len(args) == 1:
        var_in = args[0]
    else:
        var_in = args

    if var_in is None:
        return []
    elif isinstance(var_in, list):
        return var_in
    elif isinstance(var_in, tuple) or isinstance(var_in, set):
        return list(var_in)
    else:
        return [var_in]


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
