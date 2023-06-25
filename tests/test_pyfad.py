import ast
import sys
import unittest
import math
from itertools import chain

import pyfad
from .examples import fxyz, fx


fdH = 1e-8
tolFD = fdH * 10
tolAD = 1e-14


def almostEqFD(r1, r2):
    return almostEq(r1, r2, tolFD)


def almostEq(r1, r2, tol=tolAD):
    d = relNormMax(r1, r2)
    print('rel error', d)
    return d < tol


class WrongDerivative(BaseException):
    pass


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

class Pyfad(unittest.TestCase):

    def assertEqFD(self, f, r1, r2):
        if not almostEqFD(r1, r2):
            raise(WrongDerivative(r1, r2))
        return True


    def checkDer(self, func, args, dx, seed=1, active=[]):
        (der, r) = pyfad.DiffFD(func, *args, seed=seed, active=active, h=fdH)
        print('cd', (der, dx))
        self.assertTrue(self.assertEqFD(func, der, dx))

    def test_D_f1(self):
        df = pyfad.D(f1)
        print('test f1', df)

    def test_Diff_f1(self):
        df = pyfad.Diff(f1)
        print('test2', df)

    axyz = ['x', 'y', 'z']
    def test_D_f1_active(self):
        df = pyfad.D(f1, active=self.axyz)
        print('LOCALS', pyfad.locals(f1))
        print('test f1', df)
        print('test f1 f', pyfad.py(f1))
        print('test f1 d/dx f', pyfad.Dpy(f1))

    def do_call_xyz(self, func, args):
        y = func(*args)
        df = pyfad.D(func, active=self.axyz)
        print('df', df)
        args = list(args)
        print('dargs', list(pyfad.czip([1, 0, 0], args)))
        dydx, y = df(*pyfad.czip([1, 0, 0], args))
        dydy, y = df(*pyfad.czip([0, 1, 0], args))
        dydz, y = df(*pyfad.czip([0, 0, 1], args))
        self.checkDer(func, args, dydx, active='x')
        self.checkDer(func, args, dydy, active='y')
        self.checkDer(func, args, dydz, active='z')
        self.checkDer(func, args, dydx, seed=[1, 0, 0])
        self.checkDer(func, args, dydy, seed=[0, 1, 0])
        self.checkDer(func, args, dydz, seed=[0, 0, 1])

    def test_D_f1_call(self):
        self.do_call_xyz(f1, [1,2,3])

    def test_D_f2_call(self):
        self.do_call_xyz(f2, [1,2,3])

    def test_D_f3_call(self):
        self.do_call_xyz(f3, [1,2,3])

    def do_sourceDiff_f_xyz(self, func, args=None):
        if args is None:
            args = [1,2,3]
        (d_r, r) = pyfad.DiffFor(func, *args)
        self.checkDer(func, args, d_r)

    def do_sourceDiff_f_x(self, func, args=None):
        if args is None:
            args = [2]
        self.do_sourceDiff_f_xyz(func, args)

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
        self.assertTrue(self.assertEqFD(f, dr, math.cos(x)))

    def test_DiffFD_cos(self):
        f = fx.fcos
        x = 2
        dr, r = pyfad.DiffFD(f, x)
        self.assertTrue(self.assertEqFD(f, dr, -math.sin(x)))
        
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

        dr_y2, r = pyfad.DiffFD(f, x, y, z, seed=[0,1,0])
        self.assertTrue(almostEq(dr_y2, dr_y))

        with self.assertRaises(IndexError):
            dr_y2, r = pyfad.DiffFD(f, x, y, z, seed=[0,1])

        dr_z, r = pyfad.DiffFD(f, x, y, z, active='fz2')
        self.assertTrue(len(dr_z) == 0)

    def test_sD_f4(self):
        self.do_sourceDiff_f_xyz(fxyz.f4)

    def test_sD_f5(self):
        self.do_sourceDiff_f_xyz(fxyz.f5)

    def test_sD_fsin(self):
        self.do_sourceDiff_f_x(fx.fsin)

    def test_sD_fcos(self):
        self.do_sourceDiff_f_x(fx.fcos)

    def test_sD_fsqrt(self):
        self.do_sourceDiff_f_x(fx.fsqrt)

    def test_sD_ftan_1(self):
        pyfad.delrule(math.tan)
        with self.assertRaises(pyfad.NoRule):
            self.do_sourceDiff_f_x(fx.ftan)
        pyfad.restorerule(math.tan)
 
    def test_sD_ftan(self):
        adf = lambda dx, x: (0, math.tan(x))
        pyfad.setrule(math.tan, adf)
        with self.assertRaises(WrongDerivative):
            self.do_sourceDiff_f_x(fx.ftan)
        pyfad.delrule(math.tan)

    def test_sD_fsqrt_1(self):
        adf = lambda dx, x: (0, math.sqrt(x))
        pyfad.setrule(fx.gbabylonian, adf)
        with self.assertRaises(WrongDerivative):
            self.do_sourceDiff_f_x(fx.fbabylonian)
        pyfad.delrule(fx.gbabylonian)

    def test_sD_fsqrt_2(self):
        adf = lambda dx, x: (0.5 * dx / math.sqrt(x), math.sqrt(x))
        pyfad.setrule(fx.gbabylonian, adf)
        self.do_sourceDiff_f_x(fx.fbabylonian)
        pyfad.delrule(fx.gbabylonian)

    def test_sD_ftan(self):
        adf = lambda dx, x: (dx / (1 + x*x), math.atan(x))
        pyfad.setrule(math.atan, adf)
        self.do_sourceDiff_f_x(fx.fatan)
        print('RULES', pyfad.getrules())
        pyfad.delrule(math.atan)

    def test_fxyz(self, module=None, args=[1,2,3]):
        if module is None:
            module = fxyz
        fnames = [f for f in dir(module) if f[0] == 'f']
        for f in fnames:
            fn = getattr(module, f)
            print(f'Test function {fn.__name__} from {module.__name__}')
            self.do_sourceDiff_f_xyz(fn, args=args)

    def test_fx(self, module=None):
        self.test_fxyz(module=fx, args=[0.234])
