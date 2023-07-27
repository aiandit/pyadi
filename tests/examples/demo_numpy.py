"""Demo demo_numpy
--------------------

This demos shows that using numpy instead of plain floats works as
well. We must however redefine our function slightly, because the stop
condition does not work with arrays. The result is
:py:func:`~.demo_numpy.gbabylonian`.

"""
import numpy
import pyadi
numpy.set_printoptions(precision=15)

def gbabylonian(x, y=1, tol=1e-7):
    if numpy.all(numpy.abs(y**2 - x) < tol):
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

        x0 = np.array([1,2,3,4,5], dtype=float)

    When calling :py:func:`.DiffFor` now, the AD function will be run
    five times automatically, and the result ``dr`` is a list of
    length five::

        dr1, r1 = pyadi.DiffFor(f, x0, verbose=0)

    We compare with :py:func:`.DiffFD` as well::

        dr_fd1, r1 = pyadi.DiffFD(f, x0)

    Printing this output would by rather lengthy, and it contains a
    lot of zeros. The function is vector valued, so it does not make
    sense to iterate over all the inputs, we can get the derivatives
    as a vector using the ``seed`` parameter::

      dr2, r2 = pyadi.DiffFor(f, x0, verbose=0, seed=[numpy.ones(5)])
      dr_fd2, r2 = pyadi.DiffFD(f, x0, seed=[numpy.ones(5)])

    The seed is a list, and each item defines a `derivative
    direction`, the AD function will run once for each item. The
    output is then::

       x0 = [16.  2.  3.  4.  5.]
       r0 = [4.000000000000051 1.414213562373095 1.732050807568877 2.
        2.23606797749979 ]
       dr = [array([0.125000000000056, 0.353553390593274, 0.288675134594813,
              0.25             , 0.223606797749979])]
       dr_fd = [array([0.125000010342546, 0.353553386567285, 0.288675128246041,
              0.250000009582863, 0.223606799742981])]


    When we activate ``verbose=1`` here, the output is something like
    this::

         x0 = [16.  2.  3.  4.  5.]
         r0 = [4.000000000000051 1.414213562373095 1.732050807568877 2.
          2.23606797749979 ]
         Load and parse module __main__ source from /home/dev/src/projects/gh/pyadi/tests/examples/demo_numpy.py: 4.1 ms
         AD function produced for __main__.fbabylonian: d_fbabylonian
         AD function produced for __main__.gbabylonian: d_gbabylonian
         Timer fbabylonian adrun: 1.55 ms
         AD factor fbabylonian: 1.55 ms / 79.87 µs = 19.43
         Timer fbabylonian adrun: 184.54 µs
         AD factor fbabylonian: 184.54 µs / 79.87 µs = 2.31
         Timer fbabylonian adrun: 152.11 µs
         AD factor fbabylonian: 152.11 µs / 79.87 µs = 1.90
         Timer fbabylonian adrun: 152.11 µs
         AD factor fbabylonian: 152.11 µs / 79.87 µs = 1.90
         Timer fbabylonian adrun: 151.16 µs
         AD factor fbabylonian: 151.16 µs / 79.87 µs = 1.89
         Timer fbabylonian adrun: 146.63 µs
         AD factor fbabylonian: 146.63 µs / 68.19 µs = 2.15
         dr = [array([0.125000000000056, 0.353553390593274, 0.288675134594813,
                0.25             , 0.223606797749979])]
         dr_fd = [array([0.125000010342546, 0.353553386567285, 0.288675128246041,
                0.250000009582863, 0.223606799742981])]


    The AD factors are a lot better than what we saw in the first
    example :py:mod:`.demo_babylonian`. From the first AD factor
    reported, which includes the differentiation of the inner function
    call to :py:func:`~.demo_numpy.gbabylonian`, we can conclude that
    the source transformation time is essentially the same. Since the
    function now takes longer, because :py:mod:`numpy` kicks in, the
    first AD factor is much lower already. Then the caching and the
    warming up leads to quite reasonable AD factors of sometimes even
    less than two. Of course this is still a small example, but this
    can be seen as an even harder case for AD, as the whole code is
    still `interpreter-bound`, the actual float ops play a very small
    role until we increase the vector length to 1000 at least.

    """
    f = fbabylonian

    x0 = numpy.array([16,2,3,4,5], dtype=float)
    r0 = f(x0)

    print(f'x0 = {x0}')
    print(f'r0 = {r0}')

    assert numpy.linalg.norm(r0 * r0 - x0) < 1e-7

    dr1, r1 = pyadi.DiffFor(f, x0, verbose=1)
    assert numpy.linalg.norm(r1 - r0) < 1e-7

    # print(f'dr = {dr1}, r = {r1}')

    dr_fd1, r1 = pyadi.DiffFD(f, x0)
    # print(f'dr_fd = {dr_fd1}, r = {r1}')

    assert all([ numpy.linalg.norm(dr_fd1[i] - dr1[i]) < 1e-7 for i in range(len(dr1)) ])


    dr2, r2 = pyadi.DiffFor(f, x0, verbose=1, seed=[numpy.ones(5)])
    assert numpy.linalg.norm(r1 - r0) < 1e-7

    print(f'dr = {dr2}')

    dr_fd2, r2 = pyadi.DiffFD(f, x0, seed=[numpy.ones(5)])
    print(f'dr_fd = {dr_fd2}')

    assert all([ numpy.linalg.norm(dr_fd2[i] - dr2[i]) < 1e-7 for i in range(len(dr2)) ])



if __name__ == "__main__":
    run()
