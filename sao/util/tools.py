def _parse_to_list(*args):
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
