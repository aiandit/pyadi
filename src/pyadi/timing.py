from .astvisitor import isbuiltin
from .timer import Timer

def decorator(catch=[], height=1,  **opts):
    """This function produces a decorator :py:func:`inner` that always
    install another layer of function calls around the result (a
    function) it receives from inner layers.

    In addtion to the parameters, there are two closure variables,
    stack and found. The function :py:func:`timing` that the decorator
    produces maintains the current stack height in stack. When the
    function name matches an entry in catch, height is set to stack.

    When found + height > stack, then the call to the function
    produced by the preceding layer is sent through a timing with
    :py:class:`.Timer`.

    """
    stack = 0
    found = 0

    def inner(done, key, f):
        """The decorator produced by :py:func:`decorator`."""

        adfun = done(key)

        def timing(*args, **kw):
            """The runtime function that the decorator
            :py:func:`inner` produces."""

            nonlocal found, stack
            stack += 1
            if f.__name__ in catch:
                found = stack

            if stack < found + height:
                with Timer(f.__qualname__, f'time-{adfun.__name__}-{found}-{stack}') as t:
                    res = adfun(*args, **kw)
            else:
                res = adfun(*args, **kw)

            stack -= 1
            return res

        return timing

    return inner

# (c) 2023 AI & IT UG
# Author: Johannes Willkomm jwillkomm@ai-and-it.de
