Basic usage
***********

Differentiation
===============

.. autofunction:: pyadi.DiffFor
.. autofunction:: pyadi.DiffFD
.. autofunction:: pyadi.DiffFDNP

.. autofunction:: pyadi.setrule
.. autofunction:: pyadi.getrule
.. autofunction:: pyadi.delrule

..
   from .pyadi import DiffFor, DiffFD, DiffFDNP
   from .pyadi import D, Diff, Dpy, diff2pys, differentiate, dargs, dzeros
   from .pyadi import nvars, varv, fill, czip, clear, NoRule
   from .pyadi import getRuleModules, getHandle, initRules
   from .runtime import dzeros, unzd, joind, unjnd, DWith
   from .rules import setrule, delrule, getrules
   from .astvisitor import py, getmodule, isbuiltin, normalize, canonicalize, NoSource
