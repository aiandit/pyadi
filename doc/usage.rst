Basic usage
***********

PyADi performs first order aitomatic differentiation via source transformation.

Differentiation
===============

The main entry point to PyADi is :py:func:`.DiffFor` which accepts a
function with positional arguments, and evalutes the derivative at the
point given by the arguments.

.. autofunction:: pyadi.DiffFor

A similar result cam be obtained for comparison with
:py:func:`.DiffFD`.

.. autofunction:: pyadi.DiffFD

When the source transformation fails with :py:exc:`.NoSource`, the
function :py:func:`.setrule` must be used, maybe together with
:py:func:`.getrule` and :py:func:`.delrule`, to install a runtime
handler for the function in question into the default rule module
:py:mod:`.forwardad`. This is a function that gets called with the
current original and derivative parameters in a list of the form
`(d_x, x, d_y, y, ...)` for a function defined as `def d(x, y,
...)`. When usign ``mode='D'`` with :py:func:`.setrule`, which is the
default, then the handler will be called as `(r, d_x, x, d_y, y,
...)`, where ``r`` is the function result.

.. autofunction:: pyadi.setrule
.. autofunction:: pyadi.getrule
.. autofunction:: pyadi.delrule
.. autoexception:: pyadi.NoSource

When no handler is found, AD will be performed behind the scenes, of
which the results can be shown by using the parameters ``verbose`` or
``dump`` with :py:func:`.DiffFor`.

The main construct is the operator :py:func:`.D`, which produces a
differentiated function for any given function.

.. autofunction:: pyadi.D

The function :py:func:`.py` produces the source code of the given
function as a string, which is done by getting the AST with
:py:func:`.getast` and unparsing it with :py:func:`.unparse`, which
uses a slightly modified version of :py:func:`astunparse`, available
at https://github.com/aiandit/astunparse/archive/refs/heads/main.zip,
that is installed automatically via the :file:`requirements.txt` file.

.. autofunction:: pyadi.py
.. autofunction:: pyadi.getast

The source transformation is implemented as a series of tree
traversals, defined by the low-level AST transforming function
:py:func:`differentiate`, and the more high-level function
:py:func:`doSourceDiff`, the result of which can be seen with
:py:func:`.Dpy`. The result is an actual AST, of type
:py:class:`astunparse-ASTNode`, which has a :py:func:`str` converter
that calls :py:func:`~astunparse.unparse`, while the :py:func:`repr`
converter calls :py:func:`astunparse.unparse2j`

.. autofunction:: pyadi.Dpy
.. autofunction:: pyadi.differentiate
.. autofunction:: pyadi.doSourceDiff

A crucial function is :py:func:`.dzeros`, it is called very often by
the differentiated code and the results of it are very important in
several respects. For example, generic objects are not cloned, but all
their fields are set to zeros. Arrays from :py:mod:`numpy` are cloned
with :py:func:`~numpy.zeros`, however. Which means, should a third
type behave as :py:func:`~numpy.zeros`, you would have to overwrite
:py:func:`dzeros` (not yet possible). It is however not required to
import :py:func:`.dzeros` into your code, and likewise not the other
runtime functions :py:func:`.unzd`, :py:func:`.joind`, and
:py:func:`.unjnd`, which, in addition to :py:func:`.D` and
:py:func:`.Dc`, the differentiated code makes use of. For this reason,
only :py:func:`.dzeros` is exported, it may be useful when calling the
functions produced by :py:func:`.D` directly.

.. autofunction:: pyadi.dzeros
.. autofunction:: pyadi.joind
.. autofunction:: pyadi.unzd
.. autofunction:: pyadi.unjnd

An important function behind the scenes is also :py:func:`.isbuiltin`,
which basically checks if source code is available for the given
object, class, or function. It does so by calling
:py:func:`.getmodule`, which returns the module file name

.. autofunction:: pyadi.isbuiltin
.. autofunction:: pyadi.getmodule

The heavy lifting for :py:func:`.getast`, in the sense that there is a
cache of Python modules, and their one, entire, master AST, and a
mechanism to return clones of given functions or classes, is done by
:py:func:`.getmoddict` and :py:func:`.resolveImports`.

.. autofunction:: pyadi.astvisitor.getmoddict
.. autofunction:: pyadi.astvisitor.resolveImports

There are a couple of helper objects like, :py:class:`.DWith` which is
a context manager that simply delegates to a tuple of context
managers.

.. autofunction:: pyadi.DWith

The function resolution in PyADi is entrirely generic, the behaviour
of :py:mod:`.forwardad` can by overwritten by any other python module,
as for example by :py:mod:`.dummyad`, where the first item of each of the
tuples being past around is just any value that works.

The resulting functions can also be augmented, or decorated, by
further rule modules that (maybe) compose an additional wrapper
function of the function that the recieve from inner layers. Such
modules are of the same structure but they do not interfere in the
actual computations, for example :py:mod:`.timing` and
:py:mod:`.trace`.


..
   from .pyadi import DiffFor, DiffFD, DiffFDNP
   from .pyadi import D, Diff, Dpy, diff2pys, differentiate, dargs, dzeros
   from .pyadi import nvars, varv, fill, czip, clear, NoRule
   from .pyadi import getRuleModules, getHandle, initRules
   from .runtime import dzeros, unzd, joind, unjnd, DWith
   from .rules import setrule, delrule, getrules
   from .astvisitor import py, getmodule, isbuiltin, normalize, canonicalize, NoSource
