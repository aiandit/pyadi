import sys
import unittest
import math
import numpy as np

import pyadi
from .examples import fxyz, fx, fgen
from .examples.fx import f1 as f1alt,  f2 as f2alt

pyadi.Debug = True

fdH = 1e-8
tolFD = fdH * 10
tolAD = 1e-14


def almostEqFD(r1, r2):
    return almostEq(r1, r2, tolFD)


def almostEq(r1, r2, tol=tolAD):
    d = relNormMax(r1, r2)
    # print('rel error', d)
    return d < tol


class WrongDerivative(BaseException): pass
class WrongResult(BaseException): pass


def sqsum(r1):
    s1 = sum([v*v for v in r1])
    return math.sqrt(s1)


def relNormMax(r1, r2):
    dv = [ a - b for a,b in zip(pyadi.varv(r1), pyadi.varv(r2)) ]
    s1 = sqsum(pyadi.varv(r1))
    s2 = sqsum(pyadi.varv(r2))
    divi = max(s1, s2)
    if divi == 0:
        divi = 1
    return sqsum(dv) / divi


def f1(x,y,z):
    r = x*y*z
    return r

def f2(x,y,z):
    r = x*y*z
    return r

def f3(x,y,z):
    r = x*f1(x*z*17,y*z*17,y*x*17)
    return r


class TestDiffFor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pyadi.initRules(rules='pyadi.trace,ad=pyadi.forwardad', verbose=True)
        # pyadi.initRules(rules='t1=pyadi.trace,t2=pyadi.trace,t3=pyadi.trace,ad=pyadi.forwardad')
        pyadi.initRules(rules='ad=pyadi.forwardad')
        pyadi.clear()
        cls.verbose = 0
        cls.dump = 0
        cls.opts = {'dump': cls.dump, 'verbose': cls.verbose}

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
        self.assertTrue(sqsum(pyadi.varv(res)) != 0)
        self.assertTrue(self.assertEq(func, func(*args), res))

    def test_DiffFor_sin(self):
        f = fx.fsin
        x = 2
        dr, r = pyadi.DiffFor(f, x, verbose=self.verbose, dump=self.dump)
        self.assertTrue(self.assertEqFD(f, dr, math.cos(x)))

    def test_DiffFor_cos(self):
        f = fx.fcos
        x = 2
        dr, r = pyadi.DiffFor(f, x, verbose=self.verbose, dump=self.dump)
        self.assertTrue(self.assertEqFD(f, dr, -math.sin(x)))

    def test_DiffFor_partial(self):
        f = fxyz.f4
        x,y,z = 1,0.2,0.3
        dr, r = pyadi.DiffFor(f, x, y, z, **self.opts)
        dr_x, r = pyadi.DiffFor(f, x, y, z, active='x', **self.opts)
        dr_y, r = pyadi.DiffFor(f, x, y, z, active='y', **self.opts)
        dr_z, r = pyadi.DiffFor(f, x, y, z, active='z', **self.opts)
        self.assertTrue(almostEq(dr[0], dr_x))
        self.assertTrue(almostEq(dr[1], dr_y))
        self.assertTrue(almostEq(dr[2], dr_z))

        dr_y2, r = pyadi.DiffFor(f, x, y, z, seed=[ [0,1,0] ], **self.opts)
        self.assertTrue(almostEq(dr_y2, dr_y))

        with self.assertRaises(RuntimeError):
            dr_y2, r = pyadi.DiffFor(f, x, y, z, seed=[ [0,1] ], **self.opts)

        dr_z, r = pyadi.DiffFor(f, x, y, z, active='fz2', **self.opts)
        self.assertTrue(len(dr_z) == 0)

    def test_DiffFor_partial2(self):
        f = fxyz.f4
        x,y,z = 1,0.2,0.3
        dr, r = pyadi.DiffFor(f, x, y, z)
        dr_x, r = pyadi.DiffFor(f, x, y, z, active=['x'], **self.opts)
        dr_y, r = pyadi.DiffFor(f, x, y, z, active=['y'], **self.opts)
        dr_z, r = pyadi.DiffFor(f, x, y, z, active=['z'], **self.opts)
        self.assertTrue(almostEq(dr[0], dr_x))
        self.assertTrue(almostEq(dr[1], dr_y))
        self.assertTrue(almostEq(dr[2], dr_z))

        dr_y2, r = pyadi.DiffFor(f, x, y, z, seed=[ [0,1,0] ], **self.opts)
        self.assertTrue(almostEq(dr_y2, dr_y))

    def test_DiffFor_partial3(self):
        f = fxyz.f4
        x,y,z = 1,0.2,0.3
        dr, r = pyadi.DiffFor(f, x, y, z)
        dr_x, r = pyadi.DiffFor(f, x, y, z, active=[0], **self.opts)
        dr_y, r = pyadi.DiffFor(f, x, y, z, active=[1], **self.opts)
        dr_z, r = pyadi.DiffFor(f, x, y, z, active=[2], **self.opts)
        self.assertTrue(almostEq(dr[0], dr_x))
        self.assertTrue(almostEq(dr[1], dr_y))
        self.assertTrue(almostEq(dr[2], dr_z))

        dr_y2, r = pyadi.DiffFor(f, x, y, z, seed=[ [0,1,0] ], **self.opts)
        self.assertTrue(almostEq(dr_y2, dr_y))

    def test_DiffFor_partial4(self):
        f = fxyz.f4

        x,y,z = 1,2,3
        v = [x, y, z]
        dx = [1.1, -1.2, 2]

        def dft(t):
            return f(*[ v[i] + t*dx[i] for i in range(len(v)) ])

        dr, r = pyadi.DiffFor(dft, 0.0, **self.opts)

        dr_x, r = pyadi.DiffFor(f, x, y, z, active=[0], **self.opts)
        dr_y, r = pyadi.DiffFor(f, x, y, z, active=[1], **self.opts)
        dr_z, r = pyadi.DiffFor(f, x, y, z, active=[2], **self.opts)

        dr_dxc = dr_x[0]*dx[0] +  dr_y[0]*dx[1] + dr_z[0]*dx[2]

        dr_dx2, r = pyadi.DiffFor(f, x, y, z, seed=[ dx ], **self.opts)
        #print('r2', dr_dx2, dr)
        self.assertTrue(almostEq(dr_dx2, dr))

        #print('rc', dr[0], dr_dxc)
        self.assertTrue(almostEq(dr[0], dr_dxc))

    def test_DiffFor_partial5(self):
        f = fxyz.f4a

        x,y,z = 1,2,3
        v = [x, y, z]
        dx = [1.1, -1.2, 2]

        def dft(t):
            return f(*[ v[i] + t*dx[i] for i in range(len(v)) ])

        dr, r = pyadi.DiffFor(dft, 0, **self.opts)

        dr_x, r = pyadi.DiffFor(f, x, y, z, active=[0], **self.opts)
        dr_y, r = pyadi.DiffFor(f, x, y, z, active=[1], **self.opts)
        dr_z, r = pyadi.DiffFor(f, x, y, z, active=[2], **self.opts)

        dr_dxc = dr_x[0]*dx[0] +  dr_y[0]*dx[1] + dr_z[0]*dx[2]

        dr_dx2, r = pyadi.DiffFor(f, x, y, z, seed=[ dx ], **self.opts)
        #print('r2', dr_dx2, dr)
        self.assertTrue(almostEq(dr_dx2, dr))

        #print('rc', dr[0], dr_dxc)
        self.assertTrue(almostEq(dr[0], dr_dxc))

    def test_DiffFor_kw(self):
        f = fx.gbabylonian

        x = 16
        dx = [1]

        dr1, r1 = pyadi.DiffFor(f, x, f_kw=dict(tol=1e-4), **self.opts)

        dr2, r2 = pyadi.DiffFor(f, x, f_kw=dict(tol=1e-15), **self.opts)

        self.assertFalse(almostEq(dr1, dr2, tol=1e-7))
        self.assertFalse(almostEq(dr1, dr2, tol=1e-7))

    def test_DiffFor_deco(self):
        f = fx.gdeco

        x = [1,2,3]
        dx = [1,2,3]

        dr1, r1 = pyadi.DiffFor(f, x, **self.opts)

        dr2, r2 = pyadi.DiffFD(f, x, **self.opts)

        assert r1 == f(x)

        assert almostEqFD(dr1, dr2)
