
def mkActArgFunction(f, args, inds):
    """Create an inner function of only the arguments given by
    ``inds``, filling in the remaining ones statically in each call.

    Return the inner function and the remaining arguments.

    This function is differentiated automatically by
    :py:func:`.DiffFor` and :py:func:`.DiffFD` when active arguments
    were specified.

    Parameters
    ----------

    function : function
      A function of ``args``, for which an inner function is created
      of only ``[args[i] for i in inds]``

    Returns
    -------
    function, list
      A tuple of the inner function and the remaining arguments.

    """
    def inner(*actargs):
        fullargs = list(args)
        for i, k in enumerate(inds):
            fullargs[k] = actargs[i]
        return f(*fullargs)

    actargs = [args[i] for i in inds]

    return inner, actargs

def mkKwFunction(f, f_kw):
    """Create an inner function that adds ``**f_kw`` to the call to
    ``f``.

    This function is differentiated automatically by
    :py:func:`.DiffFor` when the option ``f_kw`` is used.

    Parameters
    ----------

    function : function
      A function of ``args``, for which an inner function is created
      of only ``[args[i] for i in inds]``

    kw : dict
      Dictionary of keyword parameters to be added to the call.

    Returns
    -------
    function
      A tuple of the inner function and the remaining arguments.

    """
    def inner(*args):
        return f(*args, **f_kw)

    return inner
