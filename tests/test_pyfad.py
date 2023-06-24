import ast
import sys
import unittest

import pyfad

def f1(x,y,z):
    r = x*y*z
    return r

def f2(x,y,z):
    r = x*y*z
    return r

def f3(x,y,z):
    r = x*f1(x*z*17,y*z*17,y*x*17)
    return r

class UnparseTestCase(unittest.TestCase):
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
