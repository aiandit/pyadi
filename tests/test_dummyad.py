import ast
import sys
import unittest
import math
from itertools import chain

import pyadi
from .examples import fxyz, fx, fgen

from . import test_pyadi

class TestPyfDummyad(test_pyadi.TestPyfad):

    @classmethod
    def setUpClass(cls):
        # pyadi.initRules(rules='trace,ad', verbose=True)
        # pyadi.initRules(rules='trace,ad')
        pyadi.clear()
        pyadi.initRules(rules='pyadi.dummyad')
        cls.verbose = 0
        cls.dump = 0
        cls.opts = {}

    def checkDer(self, func, args, dx, seed=1, active=[]):
        self.assertTrue(True)

    def test_sD_fsqrt_1(self): pass
    def test_sD_ftan_1(self): pass
    def test_sD_ftan_2(self): pass
