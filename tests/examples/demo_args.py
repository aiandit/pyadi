"""
Demo demo_args
==============

This demo module shows how to pass additional parameters to the
function while differentiating only w.r.t. the first.

"""

import numpy
import pyadi

from .demo_babylonian import gbabylonian as f

def run():
    """The target function is f, alias :py:func:`.gbabylonian`.

    This function has two parameters with defaults, so it can be
    called with one, two, or three positional arguments.

    Two things are important to remember here:

      - parameters with defaults are often passed via keyword
        arguments. This however does not work with the entry point
        function :py:func:`.DiffFor`, as it will consider all keyword
        arguments options. So don't do this::

          # wrong: tol is now an option to DiffFor
          dr1, r1 = pyadi.DiffFor(f, x0, tol=1e-15, active=[0])

        In this case, we want to set the ``tol`` parameter, which is
        the third, so we also have to pass the second::

          # correct: use only positional arguments
          dr1, r1 = pyadi.DiffFor(f, x0, 1, 1e-15, active=[0])

      - we want to continue to differentiate w.r.t. to ``x`` only, so
        we use the ``active`` option, as seen above.

    Note that :py:func:`.gbabylonian` itself does use keyword
    arguments. It is just the entry point function :py:func:`.DiffFor`
    that does not support them.

    Running this function produces the following output::

        x0 = 12.4
        dr = [0.1419904585617669], r = 3.5213633723318023
        dr_fd = [0.1419904638311209], r = 3.5213633723318023
        F error: 3.552713678800501e-15
        AD/FD error: 5.269354008685667e-09
        dr = [0.14199045856176618], r = 3.521363372331802
        dr_fd = [0.1419904638311209], r = 3.521363372331802
        F error: 0.0
        AD/FD error: 5.269354730330633e-09

    As we see, the function result gets better by the tighter
    tolerance, but the AD vs. FD error remains the same.

    """

    x0 = 12.4
    r0 = f(x0)

    print(f'x0 = {x0}')

    assert numpy.linalg.norm(r0 * r0 - x0) < 1e-7

    dr1, r1 = pyadi.DiffFor(f, x0, verbose=0)
    print(f'dr = {dr1}, r = {r1}')

    dr_fd1, r1 = pyadi.DiffFD(f, x0, verbose=0)
    print(f'dr_fd = {dr_fd1}, r = {r1}')

    err = numpy.linalg.norm(dr_fd1[0] - dr1[0])
    assert err < 1e-7
    print(f'F error: {abs(r1 * r1 - x0)}')
    print(f'AD/FD error: {err}')

    dr1, r1 = pyadi.DiffFor(f, x0, 1, 1e-15, verbose=0, active=[0])
    print(f'dr = {dr1}, r = {r1}')

    dr_fd1, r1 = pyadi.DiffFD(f, x0, 1, 1e-15, verbose=0, active=[0])
    print(f'dr_fd = {dr_fd1}, r = {r1}')

    err = numpy.linalg.norm(dr_fd1[0] - dr1[0])
    assert err < 1e-7
    print(f'F error: {abs(r1 * r1 - x0)}')
    print(f'AD/FD error: {err}')

if __name__ == "__main__":
    run()
