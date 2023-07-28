Features
========

PyADi performs first order automatic differentiation (AD) via source
transformation for any regular Python function `f`, regular meaning
defined with ``def``.

Unless a `rule` is defined for `f` in :py:mod:`.forwardad`, or
installed with :py:func:`.setrule`, the abstract syntax tree (AST) is
obtained with :py:func:`compile`, source differentiation is performed
and the resulting function is compiled and loaded, all on the fly.

The source differentiation proceeds as usual in forward mode AD, very
much like in `ADiMat <https://ai-and-it.de/adimat>`_ for example. For
every parameter ``x`` there will be a derivative parameter ``d_x``,
which holds the derivative of ``x``, and both are any object of the
same type and inner structure. They can also be :py:mod:`numpy` arrays
of the same size. This will then be propagated in the code line by
line, for example::

   def gplus(x, y):
       s = x + y
       return s

Is differentiated to::

   def d_gplus(d_x, x, d_y, y):
       d_s = (d_x + d_y)
       s = (x + y)
       return (d_s, s)

The nice thing about source code transformation in Python is that the
differentiated code is just ordinary Python code and it basically does
the same things as the original code did. So when a piece of code or a
function is type agnostic, the differentiated code will be too. In
this example in ``gplus`` ``x`` and ``y`` can be floats, lists, or
arrays, and in any case, when ``gplus`` runs, so will ``d_gplus``.

The multiplication operators ``*`` and ``@`` are differentiated of
course with the usual arithmetic rules::

   def fmult(x):
       r = sin(x)
       s = r * x
       t = s * 2
       z = 2 * t
       return z

Is differentiated to::

     def d_fmult(d_x, x):
         (d_r, r) = D(sin)((d_x, x))
         d_s = ((d_r * x) + (r * d_x))
         s = (r * x)
         d_t = (d_s * 2)
         t = (s * 2)
         d_z = (2 * d_t)
         z = (2 * t)
         return (d_z, z)

The source differentiation `decorates` all function calls with the
operators :py:func:`.D`, and :py:func:`.Dc`, which perform the above
process, and cache the result. Thereby the source code
differentiation, or resolution to a rule, is performed on the fly
whenever a function is first touched and the process proceeds
recursively::

    def gbabylonian(x, y=1):
        if abs(y**2 - x) < 1e-7:
            return y
        else:
            r = gbabylonian(x, (y + x / y) / 2)
            return r

Is differentiated to::

    def d_gbabylonian(d_x, x, d_y=0, y=1):
        if (abs(((y ** 2) - x)) < 1e-07):
            return (d_y, y)
        else:
            d_t_c0 = ((d_x / y) - ((x * d_y) / (y ** 2)))
            t_c0 = (x / y)
            d_t_c1 = (d_y + d_t_c0)
            t_c1 = (y + t_c0)
            (d_r, r) = Dc((d_gbabylonian, gbabylonian))(
                           (d_x, x), ((d_t_c1 / 2), (t_c1 / 2)))
            return (d_r, r)

In the case of a recursive function, as in this case ``gbabylonian``,
the recursive call can even pass the differentiated function directly
to the :py:func:`.Dc` operator, which, after one small further step,
can proceed to call it directly.

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

PyADi is well-suited both for practical and educational purposes. The
performance is quite good in our experience, it is relatively easy to
set up and use, it is applicable to a large portion of the Python
language and the differentiated code it produces can be displayed and
inspected. The set of Python builtin functions covered by rules in
:py:mod:`.forwardad` is quite small as of yet, since we are in an
early stage of decelopment.
