import ast
import sys
import unittest
import math
from itertools import chain
import threading
import time

import requests

import pyfad
from .examples import fxyz, fx, fgen, ftrace

def mkreq(x):

    re = requests.Request('https://example.com')
    re.headers['X-Test'] = f'{x}'
#    re.prepare()

    print(re)

    return x+2

class TestPyTracer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('SETUP Class')
        pyfad.initRules(rules='pyfad.trace,pyfad.dummyad,tr2=pyfad.trace', tracecalls=True)
        cls.handle_ = pyfad.getHandle('pyfad.trace')
        cls.handle = lambda x, *args, **kw: cls.handle_(*args, **kw)

        cls.handle2_ = pyfad.getHandle('tr2')
        cls.handle2 = lambda x, *args, **kw: cls.handle2_(*args, **kw)

    def do_sourceDiff_f_xyz(self, func, args=None, **kw):
        if args is None:
            args = [1,2,3]
        (d_r, r) = pyfad.DiffFor(func, *args, **kw)
        return (d_r, r)

    def test_tr_calll2(self):
        self.do_sourceDiff_f_xyz(fx.fcalll2, args=[0.234])

    def test_tr_calll4(self):
        self.do_sourceDiff_f_xyz(fx.fcalll4, args=[0.234])
    def test_tr_calll5(self):
        self.do_sourceDiff_f_xyz(fx.fcalll5, args=[0.234])

    def _test_tr_req(self):
        self.do_sourceDiff_f_xyz(mkreq, args=[0.234])

    def test_tr_hist(self):
        dres, res = self.do_sourceDiff_f_xyz(fx.fcalll5, args=[0.234], tracecalls=True)
        hist = pyfad.getHandle('pyfad.trace')(get='hist')
        hist2 = pyfad.getHandle('tr2')(get='hist')
        h1a = [h for h in hist if h not in ['range', 'len', 'sum']]
        self.assertEqual(h1a, hist2)

    def test_tr_hist2(self):
        dres, res = self.do_sourceDiff_f_xyz(fx.fcalll5, args=[0.234], tracecalls=True)
        hist1 = pyfad.getHandle('pyfad.trace')(get='hist')
        hist2 = pyfad.getHandle('tr2')(get='hist')

        dres, res = self.do_sourceDiff_f_xyz(fx.fcalll5, args=[0.234], tracecalls=True)
        hist1a = pyfad.getHandle('pyfad.trace')(get='hist')
        hist2a = pyfad.getHandle('tr2')(get='hist')

        self.assertEqual(hist1, hist1a)
        self.assertEqual(hist2, hist2a)

    def test_tr_verbose(self):
        hist = pyfad.getHandle('pyfad.trace')(get='hist')
        print('** Get trace history: ', hist)
        dres, res = self.do_sourceDiff_f_xyz(fx.fcalll5, args=[0.234], verbose=True)
        hist = pyfad.getHandle('pyfad.trace')(get='hist')
        print('Get trace history: ', hist)

    def startStopExecution(self, handle, cvsleep, delay=1e-1):
        while True:
            # lock is always owned by the program here
            # command sleep
            handle(condition=cvsleep, cmd='pause')

            # wait until the program actually sleeps
            cvsleep.acquire()

            # let program wait
            time.sleep(delay)

            # delete condition and cmd from control dict
            handle('condition', 'cmd')

            # notify and release lock, program acquires lock and exits wait
            cvsleep.notify()
            cvsleep.release()

            # let program work
            time.sleep(1e-3)

            # finished? exit thread
            if handle(get='done'):
                break

    def test_tr_control(self):
        cvsleep = threading.Condition()
        cvsleep.acquire()
        self.handle(done=False)
        tr = threading.Thread(target=self.startStopExecution,
                              args=(self.handle, cvsleep, 8e-3))
        tr.start()
        time.sleep(1)
        dres, res = self.do_sourceDiff_f_xyz(fx.flong, args=[0.234])
        hist = self.handle(get='hist')
        self.handle(done=True)
        cvsleep.release()
        tr.join()
        print('Start/Stop thread joined, finish.')

    def _test_tr_loadast(self):

        print('Test function', ftrace.floadast(0))
        print('Test gunction', ftrace.gunparse(ftrace.floadast(0)))

        self.do_sourceDiff_f_xyz(ftrace.floadast, args=[0.234])
        self.do_sourceDiff_f_xyz(ftrace.floadast, args=[0.234])

    def _test_tr_generic(self):
        self.do_sourceDiff_f_xyz(ftrace.pargs, args=[0.234])
