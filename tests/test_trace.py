import ast
import sys
import unittest
import math
from itertools import chain
import threading
import time

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

    def setUp(self):
        print('SETUP')
        pyfad.initRules(rules='trace,dummy,tr2=trace', tracecalls=True)
        self.handle = pyfad.getHandle('pyfad.trace')
        handle2 = pyfad.getHandle('tr2')

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
        dres, res, handle = self.do_sourceDiff_f_xyz(fx.fcalll5, args=[0.234], tracecalls=True)
        hist = handle(get='hist')
        print('Get trace history: ', hist)

    def startStopExecution(self, handle, cvsleep, delay=1e-1):
        while True:
            # lock is alawys owned by the program here
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
                              args=(self.handle, cvsleep, 1e-1))
        tr.start()
        time.sleep(1)
        dres, res = self.do_sourceDiff_f_xyz(fx.flong, args=[0.234], tracecalls=True)
        hist = self.handle(get='hist')
        print('Get trace history: ', hist)
        self.handle(done=True)
        cvsleep.release()
        tr.join()
        print('Start/Stop thread joined, finish.')
