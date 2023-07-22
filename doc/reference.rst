Module and function reference
*****************************

Main module pyadi
=================

The main module contains the most important function for users of
PyADi: :py:func:`.DiffFor`, which evaluates derivatives.

It also contains the runtime function decorators :py:func:`.D` and
:py:func:`.Dc`, which are the main workhorse of PyADi. The module also
contains the mechanism by which these decorators invoke the configured
rule modules, which by default is just :py:mod:`.forwardad`. When the
rule modules at last ask the core to handle the function, source
transformation is used to produce a differentiated function. The main
module also contains the main AST visistor that performs the source
code differentiation, while many tools related to that task are also
in :py:mod:`.astvisitor`.

.. automodule:: pyadi.pyadi
   :members:
   :undoc-members:
   :special-members: __init__, __call__, __str__, __repr__, __iter__, __next__, __enter__, __exit__


Runtime rule modules
====================

The rule modules are initialized by :py:func:`.initRules`, which takes
a list of python module names and imports them. PyADi provides some
builtin rule modules which are documented in this section.

Any python module may be used as a rule module, as long as it contains
a function :py:func:`~.forwardad.decorator`.

The rules modules come in two categories: Modules that do the actual
thing, of which there are :py:mod:`.forwardad` and :py:mod:`.dummyad`,
and modules which do nothing, inserting themselves into the chain
without changing anything to the arguments and results. They may
choose to intercept all the function calls, but then again they do not
change the result and call the given function. The No-Op modules are
:py:mod:`.trace` and :py:mod:`.timing`,

Module forwardad
----------------

.. automodule:: pyadi.forwardad
   :members:
   :undoc-members:


Module dummyad
--------------

This rule module is an attempt to provide a set of rules that makes
the differentiated code run, with no regard for the derivative result.

.. automodule:: pyadi.dummyad
   :members:
   :undoc-members:

Module trace
------------

This rule module adds a call to any function call that monitors for
certain tasks, like printing the function name and possibly arguments,
and for other tasks such as even interrupting the program.

.. automodule:: pyadi.trace
   :members:
   :undoc-members:

Module timing
-------------

This rule module adds a call to any function call that monitors
certain function calls. When a matching function is caught its call
and its descendants for a configurable height are timed with
:py:class:`.Timer`.

.. automodule:: pyadi.timing
   :members:
   :undoc-members:


AST utility modules
===================

Module astvisitor
-----------------

.. automodule:: pyadi.astvisitor
   :members:
   :undoc-members:
   :special-members: __init__, __call__, __str__, __repr__, __iter__, __next__, __enter__, __exit__

Module nodes
------------

.. automodule:: pyadi.nodes
   :members:
   :undoc-members:
   :special-members: __init__, __call__, __str__, __repr__

Other utility modules
=====================

Module runtime
--------------

.. automodule:: pyadi.runtime
   :members:
   :undoc-members:
   :special-members: __init__, __call__, __str__, __repr__


Module timer
------------

.. automodule:: pyadi.timer
   :members:
   :undoc-members:
   :special-members: __init__, __call__, __str__, __repr__

Module cmdline
--------------
.. automodule:: pyadi.cmdline
   :members:
   :undoc-members:
   :special-members:
