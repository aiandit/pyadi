Testing
*******

Install prerequisites
=====================

Testing PyADi requires the source archive, which can be obtained from
GitHub. In the sources, the directory tests contains the tests. The
unit testing will run all python modules beginning with "test" found
under that directory using the :py:mod:`unittest` framework.

Testing uses tox and the additional requirements listed in
test_requirements.txt. The test runner tox itself is not listed there
as it runs the tests, so we need to install tox manually::

  pip install tox

Configuring python versions
===========================

Add the python versions to be tested in tox.ini, line 2, e.g. py311
for python3.11.

Running tox
===========

Run tox::

  tox

Tox runs the tests once for each python version configured in file
tox.ini, line 2, and found on the local system. Run tox only with a
specific environment::

  tox -e py39

Tox creates virtual environments for each python version configured
and found. It installs in each environment exactly the dependencies
listed in test_requirements.txt and thus runs the tests in complete
isolation.

The generated environments are stored under the directory .tox. When a
complete refresh is desired simply delete that entire directory, but
that should be seldom necessary as tox notices changes to
test_requirements.txt::

  rm -rf .tox

Tox runs pytest and additional options for pytest can be added after
the -- option, like the option -s which will capture printed output::

  tox -- -s

To stop tests after the first error, because otherwise output can be
quite difficult to read, use option -x::

  tox -- -s -x

To filter tests by class name, module name or function name, use
option -k::

  tox -- -s -x -k TestPyADi

To filter tests by several names, use option -k with an expression::

  tox -- -s -x -k "testpyadi and particular"

When tests get too creative with non-linearily combining floats,
floating point exceptions may occur which are reported as warnings. To
promote these to errors, and see the stack that causes them, use::

  tox -- -s -x -W error::RuntimeWarning


Tests layout and organization
=============================

All functions beginning with "f" from the test example modules
tests.examples.fx and tests.examples.fxyz are run automatically by the
test module tests.test_pyadi, so the easiest way to add tests for
new functionality is to add some function beginning with f in one of
those.

These automatically tested functions get a single float or a list of
three floats as arguments, respectively. The test function can return
whatever value, the resulting derivatives can be checked against
:py:func:`.DiffFD` in any case. The tests also check the function
result for correctness. The tests also assure that the sum of all
floats in the result is not zero, to ensure that the test function at
least does not return constant zero.

However, returning large structures from a test function can result in
complex error messages when there are problems. For this reason it is
preferable to return a single float from a test function, but that
value should ideally combine all the values that where created and
calculated in whatever form so that as many operations as possible are
covered when checking the derivative. There are already some helper
functions in the test modules that can help to squash complex data
structures into a single floats, like :py:func:`gl_sum2`, which
recursively adds all items of lists together.

There should ideally be, for each Python language construct, one small
function in tests.examples.fx demonstrating its use. When the
construct involves passing more than one argument of a float, a test
function not beginning with f should be created plus a function that
does begin with f and calls that function. The latter should construct
whatever data structures the former requires in Python code, ideally
filling all sensible float values with non-linear combinations of the
input arguments. Preferably use simple non-linear operations with
universal domain, like `*` or :py:func:`sin`.

For example, in tests.examples.fx function :py:func:`ggenerator` is a
simple generator function, and :py:func:`fgenerator` tests it by
constructing a small list of powers of x, calls the generator function
and iterates it. Since the list is plain, and PyADi already supports
:py:func:`sum`, it can use sum to collapse the resulting list back to a
single float::

  def ggenerator(l):
    for i in range(len(l)):
        yield l[i]

  def fgenerator(x):
      l = [x, x*x, x*x*x]
      vl = [v for v in ggenerator(l)]
      return sum(vl)

Then, function :py:func:`ggenerator2` gets a little bit bolder and
more creative, so it wants to be tested with a longer input list, and
function :py:func:`gl_sum2` is needed to collapse the result::

  def ggenerator2(l):
      for i in range(len(l)):
          if i == 0:
              yield l[i]
          elif i % 2 == 1:
              yield [sin(l[i]), cos(l[i])]
          else:
              yield fsin(l[i])*l[i-1] + l[i-2]


  def fgenerator2(x):
      l = [x, x*x, x*x*x]
      l = l + l + l
      vl = [v for v in ggenerator2(l)]
      return gl_sum2(vl)

All tests in tests.test_pyadi are run also by test_pyadi_repl, and
likewise for tests.test_numpy and pyadi.test_numpy_repl, via
inheritance. The difference is that the option ``replaceops`` is set
to True.

The test module tests.test_dummyad also inherits from test_pyadi, but
it uses the rule module :py:mod:`.dummyad` instead of
:py:mod:`.forwardad`. So here no sensible derivatives are computed,
and of course also not checked. The function result is checked for
correctness. Otherwise, count it as a win then the code runs. These
tests can run before those that test the actual derivatives, so when
new features are added and tested, one of these tests is likely to
fail first whenever test_pyadi would fail as well. However, it's of
course more sensible to see what goes wrong when using the real
thing. In such situations it is advisable to filter for those main
tests only::

  tox -- -s -x -k testpyadi

Top priority is of course that those run. Once they do, test_dummyad
should work too. If not, some work would be required there, but that is
not the main goal.

The test modules tests.test_difffor and tests.test_difffd specifically
test the entrypoints :py:func:`.DiffFor`, :py:func:`.DiffFD`, and
:py:func:`.DiffFDNP`.

The testmodule tests.test_trace tests more specific cases involving
:py:mod:`.trace`, in combination with :py:mod:`.dummyad`. Tests trying
to apply PyADi to generic Python code without regard for sensible
derivatives should be placed in that module.

Full code examples should be given their own module in package
tests.examples, such as :py:mod:`tests.examples.cylfit`, and dedicated
test functions calling the code and testing the derivatives should be
added to tests.test_numpy.


Running Examples
================

Some example codes can also be run directly, like the cylinder fit,
which contains functions for running the full parameter estimation
with various solvers using the PyADi derivatives::

  python tests/examples/cylfit.py

or::

  python -m tests.examples.cylfit

Some of the tests require additional packages like :py:mod:`scipy`,
:py:mod:`nlopt` or AI & IT's :py:mod:`uopt`, these dependencies are
all localized to the test functions, so you need to install only what
you want to run, e.g. to try scipy's fmin_cg with AD derivatives::

  pip install scipy
  python -m tests.examples.cylfit fmin_cg_ad

will invoke the function runfmin_cg_ad in that module.


Running Contests
================

Some example codes are benchmarked using :py:mod:`pycontest`. Each of
those is in one module tests.constest_*::

  python tests/contest_cylfit.py

or::

  python -m tests.contest_cylfit

Some benchmark results have been added to the source repo, to update
those, run the contest from the tests directory.
