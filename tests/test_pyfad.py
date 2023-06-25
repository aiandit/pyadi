import ast
import sys
import unittest
import math

import pyfad
from .examples import fxyz, fx


tolFD = 1e-6
tolAD = 1e-13


def almostEqFD(r1, r2):
    return relNormMax(r1, r2) < tolFD


def almostEq(r1, r2):
    return relNormMax(r1, r2) < tolAD


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

def checkDer(func, args, dx, seed=1, active=[]):
    (der, r) = DiffFD(func, args, seed=seed)

class Pyfad(unittest.TestCase):
    def test_D_f1(self):
        df = pyfad.D(f1)
        print('test f1', df)

    def test_Diff_f1(self):
        df = pyfad.Diff(f1)
        print('test2', df)

    axyz = ['x', 'y', 'z']
    def test_D_f1_active(self):
        df = pyfad.D(f1, opts={'active': self.axyz})
        print('LOCALS', pyfad.locals(f1))
        print('test f1', df)
        print('test f1 f', pyfad.py(f1))
        print('test f1 d/dx f', pyfad.Dpy(f1)[1])

    def do_call_xyz(self, func, args):
        y = func(*args)
        df = pyfad.D(func, opts={'active': self.axyz})
        print('df', df)
        args = list(args)
        dydx, y = df(*([1, 0, 0] + args))
        dydy, y = df(*([0, 1, 0] + args))
        dydz, y = df(*([0, 0, 1] + args))

    def test_D_f1_call(self):
        self.do_call_xyz(f1, [1,2,3])

    def test_D_f2_call(self):
        self.do_call_xyz(f2, [1,2,3])

    def test_D_f3_call(self):
        self.do_call_xyz(f3, [1,2,3])

    def do_sourceDiff_f_xyz(self, func, args=None):
        if args is None:
            args = [1,2,3]
        (d_r, r) = pyfad.DiffFor(func, args)
        print('r', r)
        [print('dr', i, d_r[i]) for i in range(len(d_r))]

    def test_sD_f1_call(self):
        self.do_sourceDiff_f_xyz(fxyz.f1)
        
    def test_sD_f2_call(self):
        self.do_sourceDiff_f_xyz(fxyz.f2)

    def test_sD_f3_call(self):
        self.do_sourceDiff_f_xyz(fxyz.f3)

    def test_varv(self):
        obj = [1,2,3,4]
        print(repr(obj), repr(pyfad.varv(obj)))
        self.assertEqual(obj, list(pyfad.varv(obj)))

        obj = 2
        print(repr(obj), repr(pyfad.varv(obj)))
        self.assertEqual([2], list(pyfad.varv(obj)))

        obj = {'a': [1,2,3], 'b': 4}
        print(repr(obj), repr(pyfad.varv(obj)))
        self.assertEqual([1,2,3,4], list(pyfad.varv(obj)))

    def test_fill(self):
        seed = list(range(100))
        obj = [1,2,3,4]
        print(repr(obj), repr(pyfad.fill(obj, seed)))
        self.assertEqual([0,1,2,3], pyfad.fill(obj, seed))

        obj = tuple([1,2,3,4])
        print(repr(obj), repr(pyfad.fill(obj, seed)))
        self.assertEqual(tuple([0,1,2,3]), pyfad.fill(obj, seed))

        obj = 2
        print(repr(obj), repr(pyfad.fill(obj, seed)))
        self.assertEqual(0, pyfad.fill(obj, seed))

        obj = {'a': [1,2,3], 'b': 4}
        print(repr(obj), repr(pyfad.fill(obj, seed)))
        self.assertEqual({'a': [0,1,2], 'b': 3}, pyfad.fill(obj, seed))

    def test_DiffFD_sin(self):
        f = fx.fsin
        x = 2
        dr, r = pyfad.DiffFD(f, x)
        self.assertTrue(almostEqFD(dr, math.cos(x)))

    def test_DiffFD_cos(self):
        f = fx.fcos
        x = 2
        dr, r = pyfad.DiffFD(f, x)
        self.assertTrue(almostEqFD(dr, -math.sin(x)))
        
    def test_DiffFD_partial(self):
        f = fxyz.f1
        x,y,z = 1,2,3
        dr, r = pyfad.DiffFD(f, x, y, z)
        dr_x, r = pyfad.DiffFD(f, x, y, z, active='x')
        dr_y, r = pyfad.DiffFD(f, x, y, z, active='y')
        dr_z, r = pyfad.DiffFD(f, x, y, z, active='z')
        self.assertTrue(almostEq(dr[0], dr_x))
        self.assertTrue(almostEq(dr[1], dr_y))
        self.assertTrue(almostEq(dr[2], dr_z))

