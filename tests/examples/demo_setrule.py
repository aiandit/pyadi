"""Demo demo_setrule
-----------------

Show usage of :py:func:`.setrule` to override a function's automatic
derivative, in this case. The same procedure must be used to provide
derivatives for function that do not have Python source code. This
concerns :py:mod:`numpy` in particular.

"""

import numpy
import pyadi

from .demo_babylonian import gbabylonian as f

def run():
    """Use :py:func:`.setrule` to override the derivative of
    :py:func:`~.demo_babylonian.gbabylonian`.

    Prints as output::

      x0 = 12.4
      dr = [0.1419904585617669], r = 3.5213633723318023
      Rule called
      dr2 = [0.14199045856176618], r = 3.5213633723318023

    """

    x0 = 12.4
    r0 = f(x0)

    print(f'x0 = {x0}')

    assert numpy.linalg.norm(r0 * r0 - x0) < 1e-7

    dr1, r1 = pyadi.DiffFor(f, x0, verbose=0)
    assert numpy.linalg.norm(r1 - r0) < 1e-7

    print(f'dr = {dr1}, r = {r1}')

    def rule(r, dx, x, *args):
        print('Rule called')
        return 0.5 * dx / r

    pyadi.setrule(f, rule)
    pyadi.clear(f)

    dr2, r2 = pyadi.DiffFor(f, x0, verbose=0)

    print(f'dr2 = {dr2}, r = {r2}')
    assert numpy.linalg.norm(dr2[0] - dr1[0]) < 1e-15


if __name__ == "__main__":
    run()
