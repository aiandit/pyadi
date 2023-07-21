Module and function reference
*****************************

Main module pyadi
=================

The main module contains the most important function for user of
PyADi: :py:func:`.DiffFor`, which evaluares derivatives.

.. automodule:: pyadi.pyadi
   :members:
   :undoc-members:


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

Module rules
^^^^^^^^^^^^

The contents of this module closely belong to :py:mod:`.forwardad`,
they manipulate the dictionary of :py:mod:`.forwardad` do add and
delete differentiation rule.

.. automodule:: pyadi.rules
   :members:
   :undoc-members:

Module dummad
-------------

This rule module is an attempt to provide a set of rules that makes
the differentiated code run, with no regard for the derivative result.

.. automodule:: pyadi.dummyad
   :members:
   :undoc-members:

Module trace
------------

This rule module adds a call to any function call that monitors for
certain tasks, like printing the function name and possibly arguments,
and to other tasks such as even interrupting the program.

.. automodule:: pyadi.trace
   :members:
   :undoc-members:

Module timing
-------------

This rule module adds a call to any function call that monitors
certain function calls. When a matching function is caught its call
and its descendants for a configurable height are times with
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

Module nodes
------------

.. automodule:: pyadi.nodes
   :members:
   :undoc-members:

Other utility modules
=====================

Module timer
------------

.. automodule:: pyadi.timer
   :members:
   :undoc-members:

Module cmdline
--------------
.. automodule:: pyadi.cmdline
   :members:
   :undoc-members:
