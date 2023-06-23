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
