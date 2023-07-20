from math import sin, cos, sqrt
from pycontest import *
import numpy
import auto_diff

import pyadi
from examples import swirl

def getTest(bfun):

    def inner(l):
        return [ bfun(v) for v in l]

    return inner

#contest(dict(sin=getTest(sin), cos=getTest(cos), sqrt=getTest(sqrt)), name="Trig. test", outdir='out')

def runPyfad(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFor(bfun, *args, timings=False, **kw)

    return inner

def runPyfad2(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFor(bfun, *args, timings=True, **kw)

    return inner

def runPyfad3(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFor(bfun, *args, timings=False, replaceops=True, **kw)

    return inner

def runauto_diff(bfun):

    def inner(*args, **kw):

        x = args[0]
        dx = kw['seed'][0]
        t = numpy.zeros((2,1))
        print('tx', t[0])
        with auto_diff.AutoDiff(t) as t:
            tx = x + t[0] * dx
            print('tx', t[0])
            r = bfun(tx)
            #print('r', r)
            y, Jf = auto_diff.get_value_and_jacobian(r)

        return Jf, y

    return inner

def runPyFD(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFD(bfun, *args, **kw)

    return inner

def runPyFDNP(bfun):

    def inner(*args, **kw):
        return pyadi.DiffFDNP(bfun, *args, **kw)

    return inner

def runF(bfun):

    def inner(*args, **kw):
        return bfun(*args)

    return inner

def mkInput(N):
    Ns = max(1, int(sqrt(N)))
    x = numpy.random.rand(Ns, Ns)
    #print('mki', x.shape, x)
    return (x,), { 'seed':  [ numpy.random.rand(Ns*Ns) ] }

def reset():
    pyadi.clear()

def ftest(x):
    #print('f', x.shape, x)
    M2 = numpy.matmul(x, 2*x)
    r = numpy.linalg.norm(numpy.diag(M2@M2))
    return r

def run():
    fds = dict(normDiag=runF(ftest),
               fd_normDiag=runPyFD(ftest), fd_normDiag2=runPyFDNP(ftest),
               ad_normDiag=runPyfad(ftest), ad_normDiag2=runPyfad2(ftest), ad_normDiag3=runPyfad3(ftest))

    args0, kw0 = mkInput(10)
    # print(args0, kw0)

    res = [ fds[k](*args0, **kw0) for k in fds ]
    # print(res)
    assert all( [ abs(res[i][0][0] - res[1][0][0]) < 1e-5 for i in range(1, len(fds)) ] )

    contest(fds,
            timeout=1, input=mkInput, reset=reset,
            name="M2normtest", title="M2, Norm Test", outdir='out',
            print=True, show=False)

if __name__ == "__main__":
    run()
