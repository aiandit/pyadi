import ast
import sys
import unittest
import math
from itertools import chain
import numpy as np

import pyfad
from .examples import fxyz, fx, fgen
from .examples.fx import f1 as f1alt,  f2 as f2alt

pyfad.Debug = True

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
    dv = [ a - b for a,b in zip(pyfad.varv(r1), pyfad.varv(r2)) ]
    s1 = sqsum(pyfad.varv(r1))
    s2 = sqsum(pyfad.varv(r2))
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


class TestDiffFD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pyfad.initRules(rules='pyfad.trace,ad=pyfad.forwardad', verbose=True)
        # pyfad.initRules(rules='t1=pyfad.trace,t2=pyfad.trace,t3=pyfad.trace,ad=pyfad.forwardad')
        pyfad.initRules(rules='ad=pyfad.forwardad')
        pyfad.clear()
        cls.verbose = 2
        cls.dump = 0
        cls.opts = {}

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

    def test_DiffFD_sin(self):
        f = fx.fsin
        x = 2
        dr, r = pyfad.DiffFD(f, x, verbose=self.verbose, dump=self.dump)
        self.assertTrue(self.assertEqFD(f, dr, math.cos(x)))

    def test_DiffFD_cos(self):
        f = fx.fcos
        x = 2
        dr, r = pyfad.DiffFD(f, x, verbose=self.verbose, dump=self.dump)
        self.assertTrue(self.assertEqFD(f, dr, -math.sin(x)))

    def test_DiffFD_partial(self):
        f = fxyz.f4
        x,y,z = 1,2,3
        dr, r = pyfad.DiffFD(f, x, y, z)
        dr_x, r = pyfad.DiffFD(f, x, y, z, active='x')
        dr_y, r = pyfad.DiffFD(f, x, y, z, active='y')
        dr_z, r = pyfad.DiffFD(f, x, y, z, active='z')
        self.assertTrue(almostEq(dr[0], dr_x))
        self.assertTrue(almostEq(dr[1], dr_y))
        self.assertTrue(almostEq(dr[2], dr_z))

        dr_y2, r = pyfad.DiffFD(f, x, y, z, seed=[ [0,1,0] ])
        self.assertTrue(almostEq(dr_y2, dr_y))

        with self.assertRaises(ValueError):
            dr_y2, r = pyfad.DiffFD(f, x, y, z, seed=[ [0,1] ])

        dr_z, r = pyfad.DiffFD(f, x, y, z, active='fz2')
        self.assertTrue(len(dr_z) == 0)

    def test_DiffFD_partial2(self):
        f = fxyz.f4
        x,y,z = 1,2,3
        dr, r = pyfad.DiffFD(f, x, y, z)
        dr_x, r = pyfad.DiffFD(f, x, y, z, active=['x'])
        dr_y, r = pyfad.DiffFD(f, x, y, z, active=['y'])
        dr_z, r = pyfad.DiffFD(f, x, y, z, active=['z'])
        self.assertTrue(almostEq(dr[0], dr_x))
        self.assertTrue(almostEq(dr[1], dr_y))
        self.assertTrue(almostEq(dr[2], dr_z))

        dr_y2, r = pyfad.DiffFD(f, x, y, z, seed=[ [0,1,0] ])
        self.assertTrue(almostEq(dr_y2, dr_y))

    def test_DiffFD_partial3(self):
        f = fxyz.f4
        x,y,z = 1,2,3
        dr, r = pyfad.DiffFD(f, x, y, z)
        dr_x, r = pyfad.DiffFD(f, x, y, z, active=[0])
        dr_y, r = pyfad.DiffFD(f, x, y, z, active=[1])
        dr_z, r = pyfad.DiffFD(f, x, y, z, active=[2])
        self.assertTrue(almostEq(dr[0], dr_x))
        self.assertTrue(almostEq(dr[1], dr_y))
        self.assertTrue(almostEq(dr[2], dr_z))

        dr_y2, r = pyfad.DiffFD(f, x, y, z, seed=[ [0,1,0] ])
        self.assertTrue(almostEq(dr_y2, dr_y))

    def test_DiffFD_partial4(self):
        f = fxyz.f4

        x,y,z = 1,2,3
        v = [x, y, z]
        dx = [1.1, -1.2, 2]

        dft = lambda t: f(*[ v[i] + t*dx[i] for i in range(len(v)) ])

        dr, r = pyfad.DiffFD(dft, 0)

        dr_x, r = pyfad.DiffFD(f, x, y, z, active=[0])
        dr_y, r = pyfad.DiffFD(f, x, y, z, active=[1])
        dr_z, r = pyfad.DiffFD(f, x, y, z, active=[2])

        dr_dxc = dr_x[0]*dx[0] +  dr_y[0]*dx[1] + dr_z[0]*dx[2]

        dr_dx2, r = pyfad.DiffFD(f, x, y, z, seed=[ dx ])
        #print('r2', dr_dx2, dr)
        self.assertTrue(almostEq(dr_dx2, dr))

        #print('rc', dr[0], dr_dxc)
        self.assertTrue(almostEqFD(dr[0], dr_dxc))

    def test_DiffFD_partial5(self):
        f = fxyz.f4a

        x,y,z = 1,2,3
        v = [x, y, z]
        dx = [1.1, -1.2, 2]

        dft = lambda t: f(*[ v[i] + t*dx[i] for i in range(len(v)) ])

        dr, r = pyfad.DiffFD(dft, 0)

        dr_x, r = pyfad.DiffFD(f, x, y, z, active=[0])
        dr_y, r = pyfad.DiffFD(f, x, y, z, active=[1])
        dr_z, r = pyfad.DiffFD(f, x, y, z, active=[2])

        dr_dxc = dr_x[0]*dx[0] +  dr_y[0]*dx[1] + dr_z[0]*dx[2]

        dr_dx2, r = pyfad.DiffFD(f, x, y, z, seed=[ dx ])
        #print('r2', dr_dx2, dr)
        self.assertTrue(almostEq(dr_dx2, dr))

        #print('rc', dr[0], dr_dxc)
        self.assertTrue(almostEq(dr[0], dr_dxc, tol=1e-4))

    def test_DiffFD_partial6(self):
        ftest = fxyz.f1

        def fnp(x):
            # print('fnp', x)
            return np.array([ftest(x[0], x[1], x[2])])

        x,y,z = 1,2,3
        v = np.array([x, y, z])
        dx = np.array([7, 8, 9])

        r = fnp(v)

        dr, r = pyfad.DiffFDNP(fnp, v)

        dft = lambda t: fnp(v + t*dx)
        self.assertTrue(almostEq(dft(0.0), r))

        dr_dx, r = pyfad.DiffFDNP(dft, np.array([0.0]))

        dr_dx2, r = pyfad.DiffFDNP(fnp, v, seed=[ dx ])

        dr_dxc = np.sum(dx * dr)

        # print('rc', dr @ dx, dr_dx, dr_dx2, dr_dxc)
        self.assertTrue(almostEqFD(dr @ dx, dr_dx))
        self.assertTrue(almostEqFD(dr @ dx, dr_dx2))
        self.assertTrue(almostEqFD(dr @ dx, dr_dxc))

    def test_DiffFD_partial7(self):
        ftest = fxyz.f4a

        def fnp(x):
            # print('fnp', x)
            return np.array([ftest(x[0], x[1], x[2])])

        x,y,z = 1,2,3
        v = np.array([x, y, z])
        dx = np.array([7, 8, 9])

        r = fnp(v)

        dr, r = pyfad.DiffFDNP(fnp, v)
        dr_ad, r = pyfad.DiffFor(fnp, v)
        self.assertTrue(almostEq(dr, dr_ad, tol=1e-2))

        def dft(t):
            return fnp(v + t*dx)

        self.assertTrue(almostEq(dft(0.0), r))

        dr_dx, r = pyfad.DiffFDNP(dft, np.array([0.0]))
        dr_dx_ad, r = pyfad.DiffFor(dft, np.array([0.0]))

        dr_dx2, r = pyfad.DiffFDNP(fnp, v, seed=[ dx ])

        dr_dxc = np.sum(dx * dr)
        self.assertTrue(almostEq(dr @ dx, dr_dxc))

        dr_dxc_ad = np.sum(dx * np.array([dr_ad[0][0], dr_ad[1][0], dr_ad[2][0]]))

        #print('rc', dr_ad, dr_dxc_ad, dr_dx, dr_dx_ad, dr_dx2, dr_dxc)

        self.assertTrue(almostEq(dr_dxc, dr_dxc_ad, tol=1e-2))

        self.assertTrue(almostEq(dr_dx_ad, dr_dxc_ad))

        self.assertTrue(almostEq(dr_dx, dr_dx2))

        #!!!
        self.assertFalse(almostEq(dr_dxc, dr_dx2))
