import ast
import sys
import unittest
import math
from itertools import chain

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


class TestPyfad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pyfad.initRules(rules='pyfad.trace,ad=pyfad.forwardad', verbose=True)
        # pyfad.initRules(rules='t1=pyfad.trace,t2=pyfad.trace,t3=pyfad.trace,ad=pyfad.forwardad')
        pyfad.initRules(rules='ad=pyfad.forwardad')
        pyfad.clear()
        cls.verbose = 0
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

    def checkDer(self, func, args, dx, seed=1, active=[], **kw):
        (der, r) = pyfad.DiffFD(func, *args, seed=seed, active=active, h=fdH, verbose=self.verbose, dump=self.dump, **self.opts)
        if self.verbose > 0:
            print('cd', (der, dx))
        self.assertTrue(self.assertEqFD(func, dx, der))

    def test_D_f1(self):
        df = pyfad.D(f1, verbose=self.verbose, dump=self.dump)
        if self.verbose > 0:
            print('test f1', df)

    def test_Diff_f1(self):
        df = pyfad.Diff(f1, verbose=self.verbose, dump=self.dump)
        if self.verbose > 0:
            print('test2', df)

    axyz = ['x', 'y', 'z']
    def test_D_f1_active(self):
        df = pyfad.D(f1, active=self.axyz, verbose=self.verbose, dump=self.dump)
        if self.verbose > 0:
            print('test f1', df)
            print('test f1 f', pyfad.py(f1))
            print('test f1 d/dx f', pyfad.Dpy(f1))

    def do_call_xyz(self, func, args):
        y = func(*args)
        df = pyfad.D(func, active=self.axyz, verbose=self.verbose, dump=self.dump)
        if self.verbose > 0:
            print('df', df)
        args = list(args)
        if self.verbose > 0:
            print('dargs', list(zip([1, 0, 0], args)))
        dydx, y = df(*zip([1, 0, 0], args))
        dydy, y = df(*zip([0, 1, 0], args))
        dydz, y = df(*zip([0, 0, 1], args))
        self.checkResult(func, args, y)
        self.checkDer(func, args, dydx, active='x')
        self.checkDer(func, args, dydy, active='y')
        self.checkDer(func, args, dydz, active='z')
        self.checkDer(func, args, dydx, seed=[ [1, 0, 0] ])
        self.checkDer(func, args, dydy, seed=[ [0, 1, 0] ])
        self.checkDer(func, args, dydz, seed=[ [0, 0, 1] ])

    def test_D_f1_call(self):
        self.do_call_xyz(f1, [1,2,3])

    def test_D_f2_call(self):
        self.do_call_xyz(f2, [1,2,3])

    def test_D_f3_call(self):
        self.do_call_xyz(f3, [1,2,3])

    def do_sourceDiff_f_xyz(self, func, args=None, **kw):
        if args is None:
            args = [0.1,0.2,0.3]
        (d_r, r) = pyfad.DiffFor(func, *args, **kw, verbose=self.verbose, dump=self.dump, **self.opts)
        self.checkDer(func, args, d_r)
        self.checkResult(func, args, r)
        return (d_r, r)

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

    def test_sD_fx_call(self):
        self.do_sourceDiff_f_xyz(fx.fdeco, args=[1.2])

    def test_varv(self):
        obj = [1,2,3,4]
        if self.verbose > 0:
            print(repr(obj), repr(pyfad.varv(obj)))
        self.assertEqual(obj, list(pyfad.varv(obj)))

        obj = 2
        if self.verbose > 0:
            print(repr(obj), repr(pyfad.varv(obj)))
        self.assertEqual([2], list(pyfad.varv(obj)))

        obj = {'a': [1,2,3], 'b': 4}
        if self.verbose > 0:
            print(repr(obj), repr(pyfad.varv(obj)))
        self.assertEqual([1,2,3,4], list(pyfad.varv(obj)))

    def test_fill(self):
        seed = list(range(100))
        obj = [1,2,3,4]
        if self.verbose > 0:
            print(repr(obj), repr(pyfad.fill(obj, seed)))
        self.assertEqual([0,1,2,3], pyfad.fill(obj, seed))

        obj = tuple([1,2,3,4])
        if self.verbose > 0:
            print(repr(obj), repr(pyfad.fill(obj, seed)))
        self.assertEqual(tuple([0,1,2,3]), pyfad.fill(obj, seed))

        obj = 2
        if self.verbose > 0:
            print(repr(obj), repr(pyfad.fill(obj, seed)))
        self.assertEqual(0, pyfad.fill(obj, seed))

        obj = {'a': [1,2,3], 'b': 4}
        if self.verbose > 0:
            print(repr(obj), repr(pyfad.fill(obj, seed)))
        self.assertEqual({'a': [0,1,2], 'b': 3}, pyfad.fill(obj, seed))

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
        pyfad.clear()
        old = pyfad.delrule(math.tan)
        with self.assertRaises(pyfad.NoRule):
            self.do_sourceDiff_f_x(fx.ftan)
        pyfad.setrule(math.tan, old)

    def test_sD_ftan_2(self):
        pyfad.clear()
        old = pyfad.delrule(math.tan)
        adf = lambda r, dx, x: 0
        pyfad.setrule(math.tan, adf)
        with self.assertRaises(WrongDerivative):
            self.do_sourceDiff_f_x(fx.ftan)
        pyfad.setrule(math.tan, old)

    def test_sD_ftan_3(self):
        pyfad.clear()
        old = pyfad.delrule(math.tan)
        adf = lambda r, dx, x: dx / (1 + x*x)
        pyfad.setrule(math.atan, adf)
        self.do_sourceDiff_f_x(fx.fatan)
        pyfad.setrule(math.tan, old)


    def test_sD_fsqrt_1(self):
        pyfad.clear()
        adf = lambda r, dx, x: 0
        pyfad.setrule(fx.gbabylonian, adf)
        with self.assertRaises(WrongDerivative):
            self.do_sourceDiff_f_x(fx.fbabylonian)
        pyfad.delrule(fx.gbabylonian)

    def test_sD_fsqrt_2(self):
        pyfad.clear()
        adf = lambda r, dx, x: 0.5 * dx / r
        pyfad.setrule(fx.gbabylonian, adf)
        self.do_sourceDiff_f_x(fx.fbabylonian)
        pyfad.delrule(fx.gbabylonian)

    def test_fxyz(self, module=None, args=[0.1,0.2,0.3]):
        if module is None:
            module = fxyz
        fnames = [f for f in dir(module) if f[0] == 'f']
        for f in fnames:
            fn = getattr(module, f)
            if self.verbose > 0:
                print(f'Test function {fn.__name__} from {module.__name__}')
            self.do_sourceDiff_f_xyz(fn, args=args)

    def test_fxyz2(self):
        self.test_fxyz(module=fxyz, args=[-0.1,0.2,0.3])

    def test_fxyz3(self):
        self.test_fxyz(module=fxyz, args=[0.1,-0.2,0.3])

    def test_fxyz4(self):
        self.test_fxyz(module=fxyz, args=[-0.1,-0.2,0.3])

    def test_fxyz5(self):
        self.test_fxyz(module=fxyz, args=[-0.1,-0.2,-0.3])

    def test_fx(self):
        self.test_fxyz(module=fx, args=[0.234])

    def test_fx2(self):
        self.test_fxyz(module=fx, args=[-0.234])

    def test_gen(self):
        self.test_fxyz(module=fgen, args=[1,2,3])

    def test_py(self, module=None):
        src = pyfad.py(fx.f1)
        self.assertEqual(src[0:10], "def f1(x):")

    def test_py2(self, module=None):
        src, imps, mods = pyfad.py(fx.f1, True)
        self.assertEqual(src[0:10], "def f1(x):")
        self.assertEqual(mods, ['math', 'm2', 'timer'])
        if self.verbose > 0:
            print(src, imps, mods)

    def test_py_meth(self, module=None):
        src, imps, mods = pyfad.py(fx.Plane.__init__, True)
        self.assertEqual(src[0:10], "def __init")
        self.assertIn('pass', src)
        self.assertNotIn('self.consumption', src)
        if self.verbose > 0:
            print(src, imps, mods)

    def test_py_meth2(self, module=None):
        src, imps, mods = pyfad.py(fx.Plane2.__init__, True)
        self.assertEqual(src[0:10], "def __init")
        self.assertIn('self.consumption', src)
        self.assertNotIn('pass', src)
        if self.verbose > 0:
            print(src, imps, mods)

    def test_ast_clearcache(self, module=None):
        pyfad.clear()

    def test_particular_fx(self):
        self.do_sourceDiff_f_xyz(fx.fkeywords4a, args=[0.234])

    def test_finner(self):
        self.do_sourceDiff_f_xyz(fx.finner, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.fuseinner, args=[0.234])

    def test_fcalll(self):
        self.do_sourceDiff_f_xyz(fx.fcalll, args=[0.234])

    def test_fcalll2(self):
        self.do_sourceDiff_f_xyz(fx.fcalll2, args=[0.234])

    def test_fprint(self):
        self.do_sourceDiff_f_xyz(fx.fprint, args=[0.234])

    def test_fprint2(self):
        self.do_sourceDiff_f_xyz(fx.fprint2, args=[0.234])

    def test_fobj(self):
        self.do_sourceDiff_f_xyz(fx.fobj, args=[0.234])

    def test_fobja(self):
        self.do_sourceDiff_f_xyz(fx.fobj, args=[0.234])

    def test_fobj2(self):
        self.do_sourceDiff_f_xyz(fx.fplane, args=[0.234])

    def test_fobj2a(self):
        self.do_sourceDiff_f_xyz(fx.fplane, args=[0.234])

    def test_fobj3(self):
        fx.fplane3(2.3)
        self.do_sourceDiff_f_xyz(fx.fplane3, args=[0.234])

    def test_flong1(self):
        self.do_sourceDiff_f_xyz(fx.flong, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.flong, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.flong, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.flong, args=[0.234])

    def test_flong2(self):
        self.do_sourceDiff_f_xyz(fx.flong2, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.flong2, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.flong2, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.flong2, args=[0.234])

    def test_flong4(self):
        self.do_sourceDiff_f_xyz(fx.flong4, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.flong4, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.flong4, args=[0.234])
        self.do_sourceDiff_f_xyz(fx.flong4, args=[0.234])

    def test_timings(self):
        self.do_sourceDiff_f_xyz(fx.flong, args=[0.234], timings=True)

    def test_tracetimings(self):
        pyfad.initRules(rules='pyfad.timing,pyfad.forwardad', catch=['flong', 'fcalllist'], height=8)
        self.do_sourceDiff_f_xyz(fx.flong, args=[0.234], timings=True)
        self.do_sourceDiff_f_xyz(fx.flong, args=[0.234], timings=True)
        self.do_sourceDiff_f_xyz(fx.flong, args=[0.234], timings=True)

        dr, r = self.do_sourceDiff_f_xyz(fx.fcalllist, args=[0.234], timings=False)
        self.assertEqual(r, fx.fsin(fx.f2(fx.f1(0.234))))
        self.do_sourceDiff_f_xyz(fx.fcalllist, args=[0.234], timings=False)
        self.do_sourceDiff_f_xyz(fx.fcalllist, args=[0.234], timings=False)

    def test_unzd(self):
        d = {'a': (1,2), 'b': (1,2), 'c': (1,2)}
        dr = { k: d[k][0] for k in d }
        r = { k: d[k][1] for k in d }
        dr2, r2 = pyfad.unzd(d)
        self.assertEqual(dr, dr2)
        self.assertEqual(r, r2)

    def test_joind(self):
        d = {'a': (1,2), 'b': (1,2), 'c': (1,2)}

        dr, r = pyfad.unzd(d)

        res = pyfad.joind([dr], [r])

        res2 = { 'd_' + k: v for k, v in dr.items() } | { k: v for k, v in r.items() }

        self.assertEqual(res, res2)

        dr2, r2 = pyfad.unjnd(res)

        self.assertEqual(dr, dr2)
        self.assertEqual(r, r2)

    def test_getSource_py(self):
        src = pyfad.py(fx.f1)
        line1 = next(l for l in src.split('\n'))
        self.assertEqual(line1, 'def f1(x):')

    def test_getSource_py_inner(self):
        inner = fx.ginner()
        src = pyfad.py(inner)
        line1 = next(l for l in src.split('\n'))
        self.assertEqual(line1, 'def inner(x, y, z):')
