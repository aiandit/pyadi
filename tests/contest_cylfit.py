from math import sin, cos, sqrt
from pycontest import *
import numpy as np

import pyadi
from examples import cylfit, cylfit2


def almostEq(r1, r2, tol=1e-14):
    d = relNormMaxNP(r1, r2)
    return d < tol


def relNormMaxNP(r1, r2):
    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)
    nd = np.linalg.norm(r1 - r2)
    divi = max(n1, n2)
    if divi == 0: divi = 1
    return nd / divi


def getTest(bfun):

    def inner(l):
        return [ bfun(v) for v in l]

    return inner

#contest(dict(sin=getTest(sin), cos=getTest(cos), sqrt=getTest(sqrt)), name="Trig. test", outdir='out')

def runPyADi(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFor(bfun, *args, timings=False, **kw)

    return inner

def runPyADi2(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFor(bfun, *args, timings=True, **kw)

    return inner

def runPyADi3(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFor(bfun, *args, timings=False, replaceops=True, **kw)

    return inner

def runPyFD(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFD(bfun, *args, **kw)

    return inner

def runPyFD(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFD(bfun, *args, **kw)

    return inner

def runF(bfun):

    def inner(*args, **kw):
        return bfun(*args)

    return inner

def mkInput(N):
    args = [np.array([0, 0, 0])]
    dataHandle(N)
    return args, { 'seed':  [ np.random.rand(3) ] }

def reset():
    pyadi.clear()

def setup():
    R0 = 1
    theta0 = 1.25 * np.pi / 180
    phi0 = 45 * np.pi / 180
    #theta0 = math.pi/4
    #phi0 = 0

    # v0 = np.array([R0 + 0.01, 0, 0])
    v0 = np.array([R0 + 1e-6, theta0, phi0])
    objComps, obj, handle = cylfit.cylfit_obj()
    obj2, handle2 = cylfit2.cylfit_obj()

    N = int(1e3)**2

    demopts = cylfit.mkCylData(N, theta0, phi0)
    demopts2 = cylfit2.mkCylData(N, theta0, phi0)

    assert almostEq(demopts, demopts2)

    handle()['points'] = demopts
    handle2()['points'] = demopts

    r0 = obj(v0)
    r02 = obj2(v0)

    assert almostEq(r0, r02)

    def setupData(N):
        print(f'Setup data {N}')
        demopts = cylfit.mkCylData(N, theta0, phi0)
        handle()['points'] = demopts
        handle2()['points'] = demopts
        print(f'Setup data {N}')

    global bfun, bfun2, dataHandle
    bfun = obj
    bfun2 = obj2
    dataHandle = setupData

dataHandle = None
bfun = None
bfun2 = None

setup()

contest(dict(cylfit=runF(bfun), ad_cylfit=runPyADi(bfun), ad2_cylfit=runPyADi2(bfun), ad3_cylfit=runPyADi3(bfun), fd_cylfit=runPyFD(bfun),
             cylfit2=runF(bfun2), ad_cylfit2=runPyADi(bfun2), fd_cylfit2=runPyFD(bfun2)),
        timeout=1e-1, input=mkInput, reset=reset,
        name="cylfit", title="Cyl. Fit Obj Function", outdir='out', print=True, show=False)
