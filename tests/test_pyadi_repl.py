import ast
import sys
import unittest
import math
from itertools import chain

import pyadi
from . import test_pyadi

class TestPyfadRepl(test_pyadi.TestPyfad):

    @classmethod
    def setUpClass(cls):
        # pyadi.initRules(rules='pyadi.trace,ad=pyadi.forwardad', verbose=True)
        # pyadi.initRules(rules='t1=pyadi.trace,t2=pyadi.trace,t3=pyadi.trace,ad=pyadi.forwardad')
        pyadi.initRules(rules='ad=pyadi.forwardad')
        pyadi.clear()
        cls.opts = {'replaceops': True}
        cls.verbose = 0
        cls.dump = 0
