"""Demo demo_directional
---------------------

Let us explore in more detail how to use the ``seed`` option and what
a `directional derivative` is, as the readers will certainly also
remember from school.

"""
import numpy
import pyadi
numpy.set_printoptions(precision=15)

from .demo_numpy import fbabylonian as f

def run():
    """The vector function :py:func:`~.demo_numpy.fbabylonian` is
    aliased to ``f`` and the starting value ``x0`` is a short vevtor::

        x0 = np.array([1,2,3,4,5], dtype=float)

    We saw in :py:mod:`.demo_numpy` that passing a `derivative
    direction` allows us to get all derivatives of this function as a
    single vector::

       dx = numpy.ones(5)

    How does this work? The AD process, running a differentiated
    function once, returns a single `directional derivative`. This is
    because we can chose one set of values for the derivative
    arguments, which are of the same type as the original
    arguments. The items in ``seed`` define those values, once for
    each direction::

      dr2, r2 = pyadi.DiffFor(f, x0, verbose=0, seed=[dx])
      dr_fd2, r2 = pyadi.DiffFD(f, x0, seed=[dx])

    The result of this is exactly a directional derivative `d/dx f(x)`
    evaluated at the point `x` along the `derivative direction` `dx`,
    which is defined as `d / dt f(x + t * dx)` evaluated at the point
    `t=0`.

    This means in particular, it can be expressed as the derivative of
    a function that has a single scalar float parameter ``t``. We can
    try this just out by defining a suitable function ``foft``::

       def foft(t):
           return f(x0 + t * dx)

    When we evaluate this function at ``t=0`` we get ``f(x)`` and when
    we evaluate the derivatives at that point we get the directional
    derivative::

        dr2, r2 = pyadi.DiffFor(foft, 0, verbose=2)
        dr_fd2, r2 = pyadi.DiffFD(foft, 0)

    PyADi will differemtiate ``foft`` to the following code, were the
    nonlocal values ``x0`` and ``dx`` are treated as having no
    derivative, which produces the calls to :py:func:`.dzeros`::

       def d_foft(d_t, t):
           d_t_c2 = ((d_t * dx) + (t * dzeros(dx)))
           t_c2 = (t * dx)
           return D(f)(((dzeros(x0) + d_t_c2), (x0 + t_c2)))

    We now have two different ways to compute the same derivatives,
    which we can compare with each other, Again, the FD method will
    inherently produce results that are only correct up to half the
    number of digits. The AD results should be correct up to the
    machine precision.

    The two FD results among each other will also be perfectly
    identical because the same float operations are performed in
    either case. For less well-behaved functions it may happen,
    however, that the directional derivative evaluated with FD is not
    the same as the full Jacobian evaluated with FD multiplied by
    ``dx``. For AD these two will of course also be exactly identical.

    """

    x0 = numpy.array([16,2,3,4,5], dtype=float)
    r0 = f(x0)

    print(f'x0 = {x0}')
    print(f'r0 = {r0}')

    dx = numpy.ones(5)
    print(f'dx = {dx}')

    assert numpy.linalg.norm(r0 * r0 - x0) < 1e-7

    dr1, r1 = pyadi.DiffFor(f, x0, verbose=0, seed=[dx])
    assert numpy.linalg.norm(r1 - r0) < 1e-7

    print(f'dr = {dr1}, r = {r1}')

    dr_fd1, r1 = pyadi.DiffFD(f, x0, seed=[dx])
    print(f'dr_fd = {dr_fd1}, r = {r1}')

    assert all([ numpy.linalg.norm(dr_fd1[i] - dr1[i]) < 1e-7 for i in range(len(dr1)) ])

    def foft(t):
        return f(x0 + t * dx)


    dr2, r2 = pyadi.DiffFor(foft, 0, verbose=0)
    assert numpy.linalg.norm(r1 - r0) < 1e-7

    print(f'dr = {dr2}')

    dr_fd2, r2 = pyadi.DiffFD(foft, 0)
    print(f'dr_fd = {dr_fd2}')

    assert all([ numpy.linalg.norm(dr_fd2[i] - dr2[i]) < 1e-7 for i in range(len(dr2)) ])
    assert all([ numpy.linalg.norm(dr2[i] - dr1[i]) < 1e-15 for i in range(len(dr1)) ])
    assert all([ numpy.linalg.norm(dr_fd2[i] - dr_fd1[i]) < 1e-15 for i in range(len(dr1)) ])


if __name__ == "__main__":
    run()
