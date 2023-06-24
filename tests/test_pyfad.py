import ast
import sys
import unittest

import pyfad

class UnparseTestCase(unittest.TestCase):
    def f1(x,y,z):
        r = x*y*z
        return r

    def test_D_f1(self):
        df = pyfad.D(self.f1)
        print('test f1', df)

    def test_Diff_f1(self):
        df = pyfad.Diff(self.f1)
        print('test2', df)

    axyz = ['x', 'y', 'z']
    def test_D_f1_active(self):
        df = pyfad.D(self.f1, opts={'active': self.axyz})
        print('LOCALS', pyfad.locals(self.f1))
        print('test f1', df)
        print('test f1 f', pyfad.py(self.f1))
        print('test f1 d/dx f', pyfad.Dpy(self.f1)[1])

    def test_D_f1_call(self):
        df, actind = pyfad.D(self.f1, opts={'active': self.axyz})
        print('actind', actind)
        print('df', df)
        dydx, y = df(*([1, 0, 0] + [1,2,3]))
        dydy, y = df(*([0, 1, 0] + [1,2,3]))
        dydz, y = df(*([0, 0, 1] + [1,2,3]))
