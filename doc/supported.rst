Supported subset of Python
==========================

The supported subset of Python covers :py:term:`functions <function>`,
:py:term:`inner functions <nested scope>`, and :std:term:`lambda
functions <lambda>`, which can have parameters, parameters with
default values, a positional wildcard or a keyword wildcard parameter
(cf. :std:token:`function definition <python-grammar:funcdef>`) and
which can be called with any combination of :py:term:`positional
arguments <positional argument>`, :py:term:`keyword arguments <keyword
argument>`, and :std:token:`starred expressions
<python-grammar:starred_and_keywords>` as Python allows.

The compound statements
:std:token:`for <python-grammar:for_stmt>`,
:std:token:`while <python-grammar:while_stmt>`,
:std:token:`if <python-grammar:if_stmt>`,
:std:token:`try <python-grammar:try_stmt>`, and
:std:token:`with <python-grammar:with_stmt>` are supported, as of
course are also
:std:token:`function definitions <python-grammar:while_stmt>`.

:std:token:`Import statements <python-grammar:import_stmt>` are
somewhat easier to handle as they are not generally processed, except
the ``from module import name`` statements.  These are are currently
processed once when a module is first loaded by :py:mod:`pyadi`,
unconditionally, meaning using ``from module import name`` within
another statement (like ``if`` or ``try``) may or not work, because
:py:mod:`pyadi` makes no attempt to find out what actually happened
when the module was loaded by Python and whether or not the import was
effective. Plain ``import`` statements need not to be processed
because :py:mod:`pyadi` finds the module from the function that is
being differentiated.

The data types :ref:`dict`, :py:func:`tuple`, and :py:func:`dict` are
supported as literals well as they are in the form of :std:token:`list
comprehensions <python-grammar:list_comprehension>` and
:std:token:`dict comprehensions <python-grammar:dict_comprehension>`,
also known as :ref:`generator expressions <tut-generators>`.

Object oriented programming with classes is supported, including
inheritance, method calling, bound methods, and the super()
function. Object methods including the constructor are differentiated.
Objects can also define a hidden __call__ method, which is also
differentiated when the object is called.

Iterators including user-defined iterators are supported but they
(i.e. the hidden methods __iter__ and __next__) are not
differentiated. However, a "derivative" iterator object will
automatically be created for each iterator object that the code uses,
and its constructor and other methods being called regularly will be
differentiated, so as long as the iterations just shuffe data around,
the derivative should also be correct.

Generator functions and the :ref:`yield statment <tut-generators>` are
also supported. Here, the generator function and the yielded
expressions are differentiated entirely.

Formatted strings are differentiated, and the rule for
:py:func:`print` prints the differentiated arguments, so the
differentiated programs will print lines with the values of
differentiated expressions, in addition to the original line.

Several of the most important functions are already supported, by it
because they are available in source of because they have been added
to the list of builtin rules in :py:mod:`.forwardad`. The latter must
happen for any function that cannot or shall not be differentiated in
source. It is a work in progress to cover more and mode builtin
functions. When a function is not covered by a rule and the source
code cannot be obtained, :py:func:`.DiffFor` will raise
:py:exc:`.NoSource`. Users can use :py:func:`.setrule` to dynamically
add rules at runtime to avoid this scenario. This can also be used to
install custom derivatives for any function.
