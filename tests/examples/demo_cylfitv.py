"""Demo demo_cylfit
================

This demo module shows how to pass differentiate a real lite function
using :py:mod:`numpy` and to perform an optimization using AI & IT's
solver usolve with the AD derivatives.

"""

import numpy as np
import pyadi
import uopt

from .cylfit import relNormMax

from .cylfit import cylfit_obj, mkCylData
#from .cylfit2 import cylfit_obj, mkCylData

def run():
    """
    """

    objComps, obj, handle = cylfit_obj()

    def fobj(x, y, udata):
        y[:] = objComps(x)

    def gobj(x, y, g, udata):
        (dr, r) = pyadi.DiffFor(objComps, x)
        y[:] = r
        for i in range(x.size):
            g[:,i] = dr[i]

    def gvobj(x, y, dx, g, udata):
        (dr, r) = pyadi.DiffFor(objComps, x, seed=[dx], timings=False)
        y[:] = r
        N, Ndd = dx.shape
        for i in range(Ndd):
            g[:,i] = dr[i]

    def gvTobj(x, y, dy, g, udata):
        def inner(x):
            return np.sum(objComps(x)*dy)
        (dr, r) = pyadi.DiffFor(inner, x, timings=False)
        y[:] = r
        for i in range(x.size):
            #print(f'r {r}, dr {dr}')
            g[:,i] = dr[i]


    R0 = 1.1
    theta0 = 0.1
    phi0 = np.pi/2

    # v0 = np.array([1, 0, 0])
    v0 = np.array([1, 0.01, 0.01])

    N = int(np.sqrt(190))**2

    demopts = mkCylData(N, R0, theta0, phi0)

    #demopts += (np.random.rand(N, 3) - 0.5) * 1e-6

    handle()['points'] = demopts

    r0 = objComps(v0)

    vs = v0.copy()
    rs = r0.copy()

    status = uopt.statusHist()

    print('start usolve')
    res = uopt.usolve(v0, vs, rs,
                      fobj, None, gvobj, gvTobj,
                      s = status,
                      opts=dict(tolGradAbs=-1, tolGradRel=-1))
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
