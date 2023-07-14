import ast
import sys
import unittest
import math
import numpy as np
from itertools import chain

import pyfad
from . import test_numpy

class TestNumpyRepl(test_numpy.TestNumpy):

    @classmethod
    def setUpClass(cls):
        # pyfad.initRules(rules='pyfad.trace,ad=pyfad.forwardad', verbose=True)
        # pyfad.initRules(rules='t1=pyfad.trace,t2=pyfad.trace,t3=pyfad.trace,ad=pyfad.forwardad')
        pyfad.initRules(rules='ad=pyfad.forwardad')
        pyfad.clear()
        cls.opts = {'replaceops': True}
        cls.verbose = 0
        cls.dump = 0
