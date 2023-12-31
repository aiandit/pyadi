from math import sin, cos, sqrt
from pycontest import *
import numpy as np

import pyadi
from examples import swirl

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
    args = swirl.initialize_starting_point(N), 1e-2
    return args, { 'seed':  [ np.random.rand(args[0].size+1) ] }

def reset():
    pyadi.clear()

bfun = swirl.swirl

contest(dict(swirl=runF(bfun), ad_swirl=runPyADi(bfun), ad_swirl2=runPyADi2(bfun), ad_swirl3=runPyADi3(bfun), fd_swirl=runPyFD(bfun)),
        timeout=1e-1, input=mkInput, reset=reset,
        name="Swirl test", outdir='out', print=True, show=False)
