import ast
import sys
import unittest

import pyfad

class UnparseTestCase(unittest.TestCase):
    def f1(x,y,z):
        r = x*y*z
        return r

    def test_function(self):
        df = pyfad.D(f1)
