"""Demo demo_cylfit
================

This demo module shows how to pass differentiate a real life function
using :py:mod:`numpy` and to perform an optimization using AI & IT's
solver `UOpt usolve <https://ai-and-it.de/uopt.html>`_ with the AD
derivatives.

"""

import numpy as np
import pyadi
import uopt

from .cylfit import relNormMax

from .cylfit import cylfit_obj, mkCylData
#from .cylfit2 import cylfit_obj, mkCylData

def run():
    """Set up the problem and the derivative functions for the solver
    and start it.

    Input values for the three parameters and the probkem size are
    set::

       R0 = 1.1
       theta0 = 0.1
       phi0 = np.pi/2

       N = int(np.sqrt(190))**2

    Sample points are created with thse values with
    :py:func:`mkCylData`.

    :py:func:`cylfit_obj` is a constructor function that returns the
    objective function and a handle function.

    The handle function is used to set the point in a variable in the
    scobe of the objective function.

    The solve needs the starting vector, the solution and the optimal
    result as :py:mod:`numpy` arrays::

      v0 = np.array([1, 0.01, 0.01])

      r0 = objComps(v0)

      vs = v0.copy()
      rs = r0.copy()

    Then the optimization can bestarted with

      res = uopt.usolve(v0, vs, rs,
                        fobj, gobj, gvobj,
                        s = status)
      print('usolve result', res, vs)


    The ``fobj``, ``gobj``, and ``gvobj`` are function handles that
    compute the function, the full Jacobian and Jacobian-vector
    products,

    """

    objComps, obj, handle = cylfit_obj()

    def fobj(x, y, udata):
        y[:] = objComps(x)

    def gobj(x, y, g, udata):
        (dr, r) = pyadi.DiffFor(objComps, x, timings=False)
        y[:] = r
        for i in range(x.size):
            g[:,i] = dr[i]

    def gvobj(x, y, dx, g, udata):
        (dr, r) = pyadi.DiffFor(objComps, x, seed=[dx], timings=False)
        y[:] = r
        N, Ndd = dx.shape
        for i in range(Ndd):
            g[:,i] = dr[i]


    R0 = 1.1
    theta0 = 0.1
    phi0 = np.pi/2

    N = int(np.sqrt(190))**2

    demopts = mkCylData(N, R0, theta0, phi0)

    #demopts += (np.random.rand(N, 3) - 0.5) * 1e-6

    handle()['points'] = demopts

    # v0 = np.array([1, 0, 0])
    v0 = np.array([1, 0.01, 0.01])

    r0 = objComps(v0)

    vs = v0.copy()
    rs = r0.copy()

    status = uopt.statusHist()

    print('start usolve')
    res = uopt.usolve(v0, vs, rs,
                      fobj, gobj, gvobj,
                      s = status)
    print('usolve result', res, vs)

    sol = vs

    sol[1] *= -1
    sol[2] *= -1

    sol[1] %= np.pi
    sol[2] %= np.pi

    rsol = obj(sol)

    errsol = relNormMax(sol, np.array([R0, theta0, phi0]))
    print(f'usolve solution {sol}, expected {[R0, theta0, phi0]}')
    print(f'usolve solution error {errsol}, final objective {rsol}')


if __name__ == "__main__":
    run()
