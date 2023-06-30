import ast
import sys
import unittest
import math
from itertools import chain

import requests

import pyfad
from .examples import fxyz, fx, fgen

def mkreq(x):

    re = requests.Request('https://example.com')
    re.headers['X-Test'] = f'{x}'
#    re.prepare()

    print(re)

    return x+2

class TestPyTracer(unittest.TestCase):
    def do_sourceDiff_f_xyz(self, func, args=None):
        if args is None:
            args = [1,2,3]
        (d_r, r) = pyfad.DiffFor(func, *args, rules='trace,dummy')

    def test_tr_calll2(self):
        self.do_sourceDiff_f_xyz(fx.fcalll2, args=[0.234])

    def test_tr_calll4(self):
        self.do_sourceDiff_f_xyz(fx.fcalll4, args=[0.234])
    def test_tr_calll5(self):
        self.do_sourceDiff_f_xyz(fx.fcalll5, args=[0.234])

    def _test_tr_req(self):
        self.do_sourceDiff_f_xyz(mkreq, args=[0.234])
