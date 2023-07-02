import ast
import sys
import unittest
import math
from itertools import chain

import pyfad
from .examples import fxyz, fx, fgen

from . import test_pyfad

class TestPyfDummyad(test_pyfad.TestPyfad):

    @classmethod
    def setUpClass(cls):
        # pyfad.initRules(rules='trace,ad', verbose=True)
        # pyfad.initRules(rules='trace,ad')
        pyfad.initRules(rules='pyfad.dummyad')

    def checkDer(self, func, args, dx, seed=1, active=[]):
        self.assertTrue(True)

    def test_sD_fsqrt_1(self): pass
    def test_sD_ftan_1(self): pass
    def test_sD_ftan_2(self): pass
