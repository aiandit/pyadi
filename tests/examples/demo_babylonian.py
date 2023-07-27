"""Demo demo_babylonian
--------------------

This module contains a function, :py:func:`.fbabylonian` which takes a
single argument ``x`` and returns the square root of ``x``, the same
as the builtin :py:func:`~math.sqrt`.

"""
import numpy
import pyadi

def gbabylonian(x, y=1, tol=1e-7):
    if abs(y**2 - x) < tol:
        return y
    else:
        r = gbabylonian(x, (y + x / y) / 2, tol=tol)
        return r

def fbabylonian(x):
    r = gbabylonian(x)
    return r

def run():
    """We alias our function to ``f`` and define a starting value ``x0``::

        f = fbabylonian

        x0 = 16

    Then we are already good to go with calling :py:func:`.DiffFor`::

        dr1, r1 = pyadi.DiffFor(f, x0, verbose=0)

        print(f'dr = {dr1}, r = {r1}')

    Since in general we will not know the correct result directly, it is
    always a good idea to compare against finite differences, at least
    once in the beginning, with :py:func:`.DiffFD`::

        dr_fd1, r1 = pyadi.DiffFD(f, x0)

        print(f'dr_fd = {dr_fd1}, r = {r1}')

    This produces the following output::

      x0 = 16
      dr = [0.12500000000005562], r = 4.000000000000051
      dr_fd = [0.12500001034254637], r = 4.000000000000051

    When getting started with a new funcion, or for deeper insights,
    it may also be interesting to set ``verbose`` to some integer >
    0. With verbose=1, information on the differentiation process is
    printed::

      x0 = 16
      Load and parse module __main__ source from /home/dev/src/projects/gh/pyadi/tests/examples/demo_babylonian.py: 3.3 ms
      AD function produced for __main__.fbabylonian: d_fbabylonian
      AD function produced for __main__.gbabylonian: d_gbabylonian
      Timer fbabylonian adrun: 1.43 ms
      AD factor fbabylonian: 1.43 ms / 5.72 µs = 249.33
      dr = [0.12500000000005562], r = 4.000000000000051
      dr_fd = [0.12500001034254637], r = 4.000000000000051
      Timer fbabylonian adrun: 60.80 µs
      AD factor fbabylonian: 60.80 µs / 4.53 µs = 13.42

    With verbose=2, the differentiated function codes will also be
    printed, which for example for our function looks like this::

      def d_fbabylonian(d_x, x):
          (d_r, r) = D(gbabylonian)((d_x, x))
          return (d_r, r)

    The function calls another function :py:func:`.gbabylonian` directly.
    The :py:func:`.D` operator will proceed to differentiate it when
    ``d_fbabylonian`` is first called, as indicated by the messages above.
    For this reason the reported runtime of the first call and the `AD
    runtime factor` are relatively large. Obviously the result of the
    process will be cached, so the subsequent call executes much faster
    already, as seen in the last line.

    The ``verbose=1`` option produces these timing messages because option
    ``timings`` defaults to ``True``. This not only to show off, but also
    to catch a common error, which is that something is not correct about
    the function, the arguments etc. These are detected by trying to run
    the function first.

    To get the best performance you should of course set ``timings`` to
    ``False``. Then, even with verbose enabled, the second call to
    :py:func:`.DiffFor` produces no output at all.

    As seen from the FD derivative values printed, this method
    inherently produces results that are only correct up to half the
    number of digits. The AD results however, should be correct up to
    the machine precision.

    """
    f = fbabylonian

    x0 = 16
    r0 = f(x0)

    print(f'x0 = {x0}')

    assert numpy.linalg.norm(r0 * r0 - x0) < 1e-7

    dr1, r1 = pyadi.DiffFor(f, x0, verbose=1)
    assert numpy.linalg.norm(r1 - r0) < 1e-7

    print(f'dr = {dr1}, r = {r1}')

    dr_fd1, r1 = pyadi.DiffFD(f, x0)
    print(f'dr_fd = {dr_fd1}, r = {r1}')

    assert all([ numpy.linalg.norm(dr_fd1[i] - dr1[i]) < 1e-7 for i in range(len(dr1)) ])

    dr2, r2 = pyadi.DiffFor(f, x0, verbose=1)
    assert all([ numpy.linalg.norm(dr2[i] - dr1[i]) < 1e-15 for i in range(len(dr1)) ])

if __name__ == "__main__":
    run()
