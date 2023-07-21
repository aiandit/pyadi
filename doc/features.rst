Features
========

PyADi can differentiate arbitray Python functions in forward mode AD.

This is done calling :py:func:`.DiffFor` with the desired function and
its arguments.

PyADi also provides two functions for finite differences, which are
useful for chacking derivatives.

PyADi works by source transformation, the source code of the function
is retrieved and differentiated into a new form that is then executed
instead. The differentiated code of a function produces a function
that returns a tuple of the derivative and the original result.

The code inside each differantiated function propagates an additional
derivative value for each orignal value in the program. Function calls
are decorated by the :py:func:`D` operator, which recursively performs
the same process for any function being called.

All the actual differentiation, except for the arithmetic operators,
happens in the module :py:mod:`.forwardad`. This module has a
mechanism to map functions being differentiated to functions in that
module which compute the required derivatives. This is required for
any builtin function, but can also be used to override the source
transformation for any function. Users can use :py:func:`.setrule` to
add rules dynamically.

PyADi provides a generic mechanism for defining how functions are
differentiated. This allows users to add own rules, augmenting the
functionality or possibly redefining the entire process currently
implemented in :py:mod:`.forwardad`. This could be used to propagate
other values than derivatives within any Python program. An example is
the :py:mod:`.dummyad` which tries to act as a replacement for
:py:mod:`.forwardad` without actually computing any derivatives.
