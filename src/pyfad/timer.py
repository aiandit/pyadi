import time

events = {}
class Timer():
    pref = ['m', 'µ', 'n', 'p']
    def __init__(self, func, kind):
        self.func = func
        self.kind = kind
        self.t0 = 0
        self.t1 = 0
        self.active = False
    def __enter__(self):
        self.t0 = time.time()
        self.active = True
    def __exit__(self, *args):
        self.t1 = time.time()
        self.active = False
        self.register()
        print(f'Timer {self.func} {self.kind}: {self}')
        if self.kind == 'adrun':
            ev = self.getev(self.func, 'run')
            if ev:
                f = self.millis() / ev['t']
                ot = self.fmt(ev['t'])
                print(f'AD factor {self.func}: {self} / {ot} = {f:.2f}')

    def millis(self):
        return 1e3*((time.time() if self.active else self.t1) - self.t0)

    def fmt(self, t):
        ms = t
        for i in range(len(self.pref)):
            p = self.pref[i]
            if ms > 1:
                break
            ms *= 1e3
        return f'{ms:.2f} {p}s'

    def __str__(self):
        return self.fmt(self.millis())

    def register(self):
        key = f'{self.func}{self.kind}'
        events[key] = {'kind': self.kind, 'func': self.func, 't': self.millis()}

    def getev(self, func, kind):
        key = f'{func}{kind}'
        try:
            return events[key]
        except:
            pass
