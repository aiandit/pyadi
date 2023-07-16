import ast
import sys
import unittest
import math
import numpy as np
from itertools import chain

import pyfad
from .examples import fxyz, fx, fgen, swirl, fnp, cylfit, cylfit2
from .examples.fx import f1 as f1alt,  f2 as f2alt

pyfad.Debug = True

fdH = 1e-8
tolFD = fdH * 10
tolAD = 1e-14


def almostEqFD(r1, r2):
    return almostEq(r1, r2, tolFD)


def almostEq(r1, r2, tol=tolAD):
    d = relNormMax(r1, r2)
    return d < tol


class WrongDerivative(BaseException): pass
class WrongResult(BaseException): pass


def sqsum(r1):
    s1 = sum([v*v for v in r1])
    return np.sqrt(s1)


def relNormMaxNP(r1, r2):
    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)
    nd = np.linalg.norm(r1 - r2)
    divi = max(n1, n2)
    if divi == 0: divi = 1
    return nd / divi

def relNormMax(r1, r2):
    if hasattr(r1, 'flat') and False:
        return relNormMaxNP(r1,r2)
    dv = [ a - b for a,b in zip(pyfad.varv(r1), pyfad.varv(r2)) ]
    s1 = sqsum(pyfad.varv(r1))
    s2 = sqsum(pyfad.varv(r2))
    divi = max(s1, s2)
    if divi == 0:
        divi = 1
    return sqsum(dv) / divi


class TestNumpy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pyfad.initRules(rules='pyfad.trace,ad=pyfad.forwardad', verbose=True)
        # pyfad.initRules(rules='t1=pyfad.trace,t2=pyfad.trace,t3=pyfad.trace,ad=pyfad.forwardad')
        pyfad.initRules(rules='ad=pyfad.forwardad')
        pyfad.clear()
        cls.opts = {}
        cls.verbose = 0
        cls.dump = 0

    def assertEq(self, f, r1, r2):
        if not almostEq(r1, r2):
            raise(WrongResult(r1, r2))
        return True

    def assertEqD(self, f, r1, r2):
        if not almostEq(r1, r2):
            raise(WrongDerivative(r1, r2))
        return True

    def assertEqFD(self, f, r1, r2):
        if not almostEqFD(r1, r2):
            raise(WrongDerivative(r1, r2))
        return True

    def checkResult(self, func, args, res):
        self.assertTrue(sqsum(pyfad.varv(res)) != 0)
        self.assertTrue(self.assertEq(func, func(*args), res))

    def checkDer(self, func, args, dx, seed=1, active=[]):
        (der, r) = pyfad.DiffFD(func, *args, seed=seed, active=active, h=fdH)
        if self.verbose > 0:
            print('cd', (der, dx))
        self.assertTrue(self.assertEqFD(func, dx, der))

    def do_sourceDiff_f_xyz(self, func, args=None, **kw):
        if args is None:
            args = [1,2,3]
        (d_r, r) = pyfad.DiffFor(func, *args, **kw, verbose=self.verbose, dump=self.dump, **self.opts)
        self.checkDer(func, args, d_r)
        self.checkResult(func, args, r)
        return (d_r, r)

    def do_sourceDiff_f_x(self, func, args=None):
        if args is None:
            args = [2]
        self.do_sourceDiff_f_xyz(func, args, **kw)

    def test_swirl(self):
        init = swirl.initialize_starting_point(1)
        seed = np.random.rand(init.size + 1)
        self.do_sourceDiff_f_xyz(swirl.swirl, args=[init, 1e-2])

    def test_fsqr(self):
        X = np.zeros((2,2))
        X.flat[:] = [1, 2, 3, 4]
        # print(f'nvars: {pyfad.nvars(X)} {X.size}')
        self.do_sourceDiff_f_xyz(fnp.gsqr, args=[X])


    def runtest_fxyz(self, module=None, args=([1,2,3],)):
        if module is None:
            module = fnp
        fnames = [f for f in dir(module) if f[0] == 'f']
        for f in fnames:
            fn = getattr(module, f)
            if self.verbose > 0:
                print(f'Test function {fn.__name__} from {module.__name__}')
            self.do_sourceDiff_f_xyz(fn, args=args)


    def test_fnp(self):
        self.runtest_fxyz(args=( [0.234, 1.234, -4.321e-3], ))


    def test_fnp2(self):
        self.runtest_fxyz(args=([0.234, 1.234, 4.321e-3], ))


    def test_cylfit(self):

        R0 = 1
        theta0 = 1.25 * math.pi / 180
        phi0 = 45 * math.pi / 180
        #theta0 = math.pi/4
        #phi0 = 0

        # v0 = np.array([R0 + 0.01, 0, 0])
        v0 = np.array([R0 + 1e-6, theta0, phi0])
        objComps, obj, handle = cylfit.cylfit_obj()
        obj2, handle2 = cylfit2.cylfit_obj()

        N = int(1e3)**2

        demopts = cylfit.mkCylData(N, R0, theta0, phi0)
        demopts2 = cylfit2.mkCylData(N, R0, theta0, phi0)

        self.assertTrue(almostEq(demopts, demopts2))

        handle()['points'] = demopts
        handle2()['points'] = demopts

        r0 = obj(v0)
        r02 = obj2(v0)
        # print('r0', r0)
        self.assertTrue(almostEq(r0, r02))

        (dr, r) = self.do_sourceDiff_f_xyz(obj, args=[v0])
        (dr, r) = self.do_sourceDiff_f_xyz(obj2, args=[v0])
        print(dr, r)
