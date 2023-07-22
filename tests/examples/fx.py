from math import sin, cos, tan, asin, acos, atan, log, exp, sqrt

import math
import math as m2

import pyadi.timer as timer

def f1(x):
    z = fsin(x)
    return z

def f2(x):
    z = fcos(x*3)
    return z

def fsin(x):
    z = sin(x)
    return z

def fcos(x):
    z = cos(x)
    return z

def ftan(x):
    z = tan(x)
    return z

def ftrig(x):
    return asin(acos(atan(tan(cos(sin(x))))))

def fsqrt(x):
    if x < 0:
        x = -x
    z = sqrt(x)
    return z

def fexp(x):
    z = exp(x)
    return z

def flog(x):
    if x < 0:
        x = -x
    z = log(x)
    return z

def fasin(x):
    z = asin(x)
    return z

def fatan(x):
    z = atan(x)
    return z

def fneg(x):
    z = -atan(x)
    return z

def fneg2(x):
    y = x*x
    z = (x + (- y))
    return z

def fdiv(x):
    r = sin(x)
    s = r / x
    t = s / 2
    z = 2 / t
    return z

def fmod(x):
    r = sin(x)
    s = r % x
    t = s % 2
    z = 2 % t
    return r + s + t + z

def fmod2(x):
    r = -sin(x)
    s = r % 3
    return s

def fpow(x):
    if x < 0:
        x = -x
    r = sin(x)
    s = r ** x
    t = r ** 2
    z = 2 ** r
    return r + s + t + z

def gbabylonian(x, y=1):
    if abs(y**2 - x) < 1e-7:
        return y
    else:
        r = gbabylonian(x, (y + x / y) / 2)
        return r

def fbabylonian(x):
    r = gbabylonian(abs(x))
    return r

def gbabylonian2(x, y=1):
    if abs(y**2 - x) < 1e-7:
        return y
    else:
        return gbabylonian(x, (y + x / y) / 2)

def fbabylonian2(x):
    return gbabylonian2(abs(x))

def ffor(x):
    l = [x, x*x, x*x*x]
    s = 0
    for v in l:
        s += v
    return s

def ffor2(x):
    l = [x, x*x, x*x*x]
    s = 0
    for i in range(len(l)):
        s += l[i]
    return s

def ffor3(x):
    l = [x, x*x, x*x*x]
    s = 0
    for (i,v) in enumerate(l):
        s += i*v
    return s

class Car:
    distance = 0
    velocity = 100

    def drive(self, t):
        self.distance += self.velocity * t

def fobj(x):
    o = Car()
    o.drive(x)
    r = o.distance
    return r

class Plane:
    distance = 0
    velocity = 100
    gas = 1e4
    consumption = 10

    def __init__(self):
        pass

    def fly(self, t):
        dist = self.velocity * t
        self.distance += dist
        self.gas -= self.consumption * dist

def fplane(x):
    o = Plane()
    o.fly(x)
    r = o.distance + o.gas
    return r

class Root:
    pass

class Plane2(Root):
    distance = 0
    velocity = 100
    gas = 1e2
    consumption = 10
    verbose = 0
    def __init__(self, c, **kw):
        self.consumption = c
        self.verbose = kw.get('verbose', 0)
        super(Plane2, self).__init__(**kw)
        if self.verbose > 0:
            print(f'Plane2.init v={self.velocity}')
    def fly(self, t):
        if self.verbose > 0:
            print(f'Plane2.fly t={t} v={self.velocity}')
        dist = self.velocity * t
        self.distance += dist
        self.gas -= self.consumption * dist

class Plane3(Plane2):
    heading = 0
    wind = 0
    def __init__(self, h, **kw):
        sinit = super(Plane3, self).__init__(**kw)
        self.heading = h
        if self.verbose:
            print(f'Plane3.init v={self.velocity}')
    def fly(self, t):
        if self.verbose:
            print(f'Plane3.fly t={t} v={self.velocity}')
        vel = self.velocity
        self.velocity += -self.wind * atan(self.heading)
        super(Plane3, self).fly(t)
        self.velocity = vel

def fplane2(x, **kw):
    y = x*x
    o = Plane2(y)
    o.fly(y)
    r = o.distance + o.gas
    if kw.get('verbose', 0) > 0:
        print('Plane2 fly dist', o.distance)
    return r


def fplane3(x, **kw):
    y = x*2
    l = [x, x*x, x*x*x ]
    o = Plane3(h=y, c=x)
    r = [ o.fly(t) for t in l ]
    r = o.distance + o.gas + o.heading + o.wind
    if kw.get('verbose', 0) > 0:
        print('Plane3 fly dist', o.distance, o.heading)
    return r

class Plane4(Plane2):
    heading = 0
    wind = 0
    def __init__(self, h, **kw):
        sinit = super().__init__(**kw)
        self.heading = h
        if self.verbose:
            print(f'Plane4.init v={self.velocity}')
    def fly(self, t):
        if self.verbose:
            print(f'Plane4.fly t={t} v={self.velocity}')
        vel = self.velocity
        self.velocity += -self.wind * atan(self.heading)
        super().fly(t)
        self.velocity = vel

def fplane4(x, **kw):
    y = x*2
    l = [x, x*x, x*x*x ]
    o = Plane4(h=y, c=x)
    r = [ o.fly(t) for t in l ]
    r = o.distance + o.gas + o.heading + o.wind
    if kw.get('verbose', 0) > 0:
        print('Plane4 fly dist', o.distance, o.heading)
    return r

class Plane5(Plane2):
    heading = 0
    wind = 0
    def __init__(this, h, **kw):
        sinit = super().__init__(**kw)
        this.heading = h
        if this.verbose:
            print(f'Plane5.init v={this.velocity}')
    def fly(this, t):
        if this.verbose:
            print(f'Plane5.fly t={t} v={this.velocity}')
        vel = this.velocity
        this.velocity += -this.wind * atan(this.heading)
        super().fly(t)
        this.velocity = vel

def fplane5(x, **kw):
    y = x*2
    l = [x, x*x, x*x*x ]
    o = Plane5(h=y, c=x)
    r = [ o.fly(t) for t in l ]
    r = o.distance + o.gas + o.heading + o.wind
    if kw.get('verbose', 0) > 0:
        print('Plane5 fly dist', o.distance, o.heading)
    return r

def gl_sum(x):
    s = 0
    for i in range(len(x)):
        s += x[i]
    return s

def gl_sum2(x):
    s = 0
    for i in range(len(x)):
        v = x[i]
        if isinstance(v, list) or isinstance(v, tuple):
            s += gl_sum2(v)
        else:
            s += v
    return s

def flist(x):
    s = gl_sum([x, x*x, x*x*x])
    return s

def flist2(x):
    s = gl_sum([f1(x)])
    return s

def flist3(x):
    l = [x, x*x, x*x*x]
    m = [f1(x) for x in l]
    s = gl_sum(m)
    return s

def flist4(x):
    s = gl_sum([x, x*x, x*x*x])
    return s

def flist5(x):
    l = [x, 2*x]
    m = [l, [3*x,4*x]]
    s = gl_sum(m[0]) + gl_sum(m[1])
    return s

def flist6(x):
    l = [f1(x), 2*x]
    m = [l, [3*x,f2(4*x)]]
    s = gl_sum(m[0]) + gl_sum(m[1])
    return s

def _flist7(x):
    l = [f1(x), 2*x]
    m = [l, [3*x,f2(4*x)]]
    s = gl_sum2(m)
    return s

def flist8(x):
    l = [f1(x), 2*x]
    m = [3*x,f2(4*x)]
    s = gl_sum2([l, m])
    return s

def flist9(x):
    l = [f1(x), 2*x]
    s = gl_sum2([l, [3*x,f1(x)]])
    return s

def flist10(x):
    l = [f1(x), 2*x, 3*x]
    m = [(v,2*v) for v in l]
    n = [f1(v[0]) for v in m]
    o = [f1(v[1]) for v in m]
    q = [[f1(v),f2(w)] for v,w in m]
    s = gl_sum2([l, m, n, o, q])
    return s

def gl_mul2(x):
    x = list(x)
    for i,v in enumerate(x):
        x[i] = x[i] * 2
    return x

def flist11(x):
    l = [f1(x), 2*x, 3*x]
    m = [(v,2*v) for v in l]
    n = [[x, 2*x, x*3], [f1(v) for v in l]]
    o = [[x, 2*x, 3*3], [f1(v) for v in gl_mul2(l)]]
    q = [[f1(v),f2(w)] for v,w in m]
    s = gl_sum2([l, m, n, o, q])
    return s

def flist12(x):
    l = [f1(x), 2*x, 3*x]
    m = [(v,2*v) for v in l]
    n = gl_sum2([[x, 2*x, x*3], [f1(v) for v in l]])
    o = gl_sum2([[x, 2*x, 3*3], [f1(v) for v in gl_mul2(l)]])
    q = gl_sum2([[f1(v),f2(w)] for v,w in m])
    s = gl_sum2([l, m])
    return s + n + o + q

def fdict2(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    s = 0
    for k, v in d.items():
        s += v
        s += d[k]
    return s

def fdict3(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    s = 0
    for k in d.keys():
        s += d[k]
    return s

def fdict4(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    d2 = {k: v*2 for k, v in d.items()}
    s = 0
    for k in d.keys():
        s += d2[k]
    return s

def fdict5(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    d2 = {k: [v*2, f1(v*v*v)] for k, v in d.items()}
    s = 0
    for k in d.keys():
        s += gl_sum(d2[k])
    return s

def fprint(x, **kw):
    l = [f1(x), 2*x, 3*x]
    if kw.get('verbose', 0) > 0:
        print(l)
    assert len(l) == 3
    if x == 0:
        raise(ValueError())
    return gl_sum2(l)

def fprint2(x, **kw):
    l = [x, x*x, x*x*x]
    if kw.get('verbose', 0) > 1:
        print(l)
        print(f'The first element of l is {l[0]}')
    for i in range(len(l)):
        if kw.get('verbose', 0) > 1:
            print(f'The {i}-th element of l is {l[i]}')
    return sum(l)

def g2(x,y): return x*y
def g2l(x,y): return x*sum(y)

def fcall(x):
    z = g2(x, x*x)
    return z

def fcallf(x):
    z = g2(x, f1(x*2))
    return z

def fcalll(x):
    z = g2l(x, [x,2*x,3,3*x])
    return z

def fcalll2(x):
    z = g2l(x, [x,2*x,f1(f2(3*x))])
    return z

def fcalll3(x):
    l = [x, x*x, x*x*x]
    A = [ [ l[i] if i == j else 0 for j in range(3) ] for i in range(3) ]
    z = g2l(gl_sum2(A), l)
    return z

def fcalll3(x):
    l = [x, x*x, x*x*x]
    A = [ [ l[i] if i == j else 0 for j in range(3) ] for i in range(3) ]
    z = g2l(gl_sum2(A), l)
    return z

def gdiag(l):
    if isinstance(l[0], list) or  isinstance(l[0], tuple):
        # is list of list == matrix, get diagonal
        return [ l[i][i] for i in range(3) ]
    # else: is list == vector, get build diagonal matrix
    return [ [ l[i] if i == j else 0 for j in range(3) ] for i in range(3) ]

def gdiag2(l):
    return [ [ l[i] for j in range(3) if i == j ] for i in range(3) ]

def fcalll3(x):
    l = [x, x*x, x*x*x]
    A = [ [ l[i] if i == j else 0 for j in range(3) ] for i in range(3) ]
    z = g2l(gl_sum2(A), l)
    return z

def fcalll4(x):
    l = [x, x*x, x*x*x]
    A = gdiag(l)
    v = gdiag(A)
    assert gl_sum2(A) == gl_sum2(v)
    assert gl_sum2(A) == gl_sum2(l)
    z = g2l(gl_sum2(A), v)
    return z

def fcalll5(x):
    l = [x, x*x, x*x*x]
    A = gdiag(gdiag(gdiag(l)))
    v = gdiag(A)
    assert gl_sum2(A) == gl_sum2(v)
    assert gl_sum2(A) == gl_sum2(l)
    z = g2l(gl_sum2(A), v)
    return z

def fa(x): return fsin(x)
def fb(x): return fsin(x)
def fc(x): return fsin(x)
def fd(x): return fsin(x)
def fe(x): return fsin(x)
def ff(x): return fsin(x)

def flong(x):
    l = [x, x*x, x*x*x]
    s = 0
    for k in range(int(1e2)):
        s += fa(fb(fc(fd(fe(ff(l[k % len(l)]))))))
    return s

def flong2(x):
    l = [x, x*x, x*x*x]
    s = 0
    s = sum([fa(fb(fc(fd(fe(ff(l[k % len(l)])))))) for k in range(int(1e2))])
    return s

def fdict(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    s = 0
    for k in d:
        s += d[k]
    return s

def gkw(a, b, **kw):
    return a*b*kw['c']

def gsumd(kw):
    if isinstance(kw, dict):
        return sum([ gsumd(v) for v in kw.values() ])
    elif isinstance(kw, list) or isinstance(kw, tuple):
        return sum([ gsumd(v) for v in kw ])
    return kw

def mkd(**kw):
    return kw

def gsumd2(a, b, **kw):
    return a*b*sum([ v for v in kw.values() ])

def fkeywords(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    s = gkw(**d)
    return s

def gmkd(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    return d

def gmkd2(x):
    d = dict(a=x, b=x*x, c=x*x*x)
    return d

def gmkdict(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    e = {'a1': x, 'b1': x*x, 'c1': x*x*x}
    return d, e

def fkeywords2(x):
    d, e = gmkdict(x)
    s = gsumd(mkd(**d, r=-x, **e))
    return s

def fkeywords3(x):
    d, e = gmkdict(x)
    s = gsumd(mkd(**d, r=f1(-x), **e))
    return s

def fkeywords3(x):
    d, e = gmkdict(x)
    s = gsumd(mkd(**d, r=f1(-x), **e))
    return s

def fkeywords4(x):
    d, e = gmkdict(x)
    s = gsumd(mkd(**d, r=gmkd(x), **e))
    return s

def fkeywords4a(x):
    d = gmkd2(x)
    e = {'x' + k: v for k, v in gmkd2(x/2).items()}
    s = gsumd(mkd(**d, r=gmkd(x), **e))
    return s

def fsin2(x):
    return math.sin(x)

def fsin3(x):
    return m2.sin(x)

def fnext(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    s = next(v for k,v in d.items() if k == 'b')
    return s

def gcall(f, x):
    return f(x)

def gmap(f, l):
    return [f(v) for v in l]

def fcalllocal(x):
    d = {'a': gcall(f1, x), 'b': gcall(f1, x*x)}
    s = next(v for k,v in d.items() if k == 'b')
    z = gcall(f1, s)
    return z

def fcalllocals(x):
    myf = f1 if x > 0 else f2
    d = {'a': gcall(myf, x), 'b': gcall(myf, x*x)}
    s = next(v for k,v in d.items() if k == 'b')
    z = gcall(myf, s)
    return z

gdefs = {'sin': sin, 'lsum': gl_sum2, 'dsum': gsumd}
def getf(name):
    return gdefs[name]

def fcallcall(x):
    myf = f1 if x > 0 else f2
    x = getf('sin')(x)
    d = {'a': gcall(myf, x), 'b': gcall(myf, x*x)}
    s = next(v for k,v in d.items() if k == 'b')
    z = gcall(myf, s)
    return z

def ginner():
    def inner(x,y,z):
        return x*y*z
    return inner

def finner(x):
    def inner(x,y,z):
        return x*y*z
    return inner(x, x*x, x*x*x)

def gmkmult(c):
    def inner(x):
        return c*x
    return inner

def fuseinner(x):
    l = [x, x*x, x*x*x]
    fs = [gmkmult(i*x) for i in range(3)]
    resl2 = [ f(l[1]) for f in fs]
    return sum(resl2)

def gprocess(x, f, done, key):
    y = f(x)
    return done(key, y)

def gcalllist(x, l):
    def step(key, res):
        if key < len(l):
            return gprocess(res, l[key], step, key+1)
        else:
            return res

    return step(0, x)

def fcalllist(x):
    fl = [f1, f2, fsin]
    return gcalllist(x, fl)

class MyIter:
    def __init__(self, l):
        self.l = l


    def apply(self, f):
        self.l = [f(v) for v in self.l]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.l):
            res = self.l[self.index]
            self.index += 1
            return res
        else:
            raise StopIteration()


def fiter(x):
    l = [x, x*x, x*x*x]
    i1 = MyIter(l)
    return sum([ v for v in i1 ])


def fiter2(x):
    l = [x, x*x, x*x*x]
    i1 = MyIter(l)
    i1.apply(fcos)
    return sum([ v for v in i1 ])


class MyClassC:
    def __init__(self, l):
        self.l = l
    def apply(self, f):
        self.l = [f(v) for v in self.l]
    def __call__(self, f):
        return sum([f(v) for v in self.l])


def fclassc(x):
    l = [x, x*x, x*x*x]
    i1 = MyClassC(l)
    r = i1(f1)
    return r


def flambda(x):
    l = [x, x*x, x*x*x]
    f = lambda x, y: x+y
    r = f(l[0], l[1]) * f(l[1], l[2])
    return r

def flambda2(x):
    l = [x, x*x, x*x*x]
    f = lambda t: t*l[2]
    r = gcall(f, f(l[0]) * f(l[1]))
    return r

def gmkobj(x):
    return Plane2(x)

def fattr(x):
    r = gmkobj(x).velocity
    return r

def fsubscript(x):
    r = gmkd(x)['c']
    return r


def fdel(x):
    d = {'a': x, 'b': x*x, 'c': x*x*x}
    s = 0
    del d['b']
    for k, v in d.items():
        s += v
        s += d[k]
    return s

def fwith(x):
    l = [x, x*x, x*x*x]
    tobj = timer.Timer(f1, 'test')
    with tobj as t:
        s = 0
        for i in range(3):
            s += l[i]
    return s

def fwith2(x):
    l = [x, x*x, x*x*x]
    with timer.Timer(f1, 'test') as t:
        s = 0
        for i in range(3):
            s += l[i]
    return s

def _fmatmul(x):
    l = [x, x*x, x*x*x]
    M = gdiag(l)
    M2 = M @ M
    return gsum2(M2)

def gtpl(a,b,c):
    return 2*a,2*b,4*c

def gtpl2(a,b,c):
    return tuple((2*a,2*b,4*c))

def glist(a,b,c):
    return [2*a,2*b,4*c]

def glist2(a,b,c):
    return list((2*a,2*b,4*c))

def fcalltpl(x):
    l = [x, x*x, x*x*x]
    r1, r2, r3 = gtpl(*l)
    return r1+r2+r3

def fcalltpl2(x):
    l = [x, x*x, x*x*x]
    r = gtpl2(*l)
    return gl_sum(r)

def fcallglist(x):
    l = [x, x*x, x*x*x]
    r1, r2, r3 = glist(*l)
    return r1+r2+r3

def fcallglist2(x):
    l = [x, x*x, x*x*x]
    r = glist2(*l)
    return gl_sum(r)

def glong3(l):
    s = 0
    for i in range(int(1e2)):
        s += gl_sum2(l)
    return s

def flong4(x):
    l = [x, x*x, x*x*x]
    M = gdiag(l)
    s = 0
    for i in range(len(l)):
        s += glong3(M)
    return s

def mydeco(f):
    def inner(*args, **kw):
        args = [ [v*2 for v in args[0]], *args[1:] ]
        res = f(*args, **kw)
        return sqrt(abs(res))
    return inner

@mydeco
def gdeco(l):
    return gl_sum(l)

def fdeco(x):
    l = [x, x*x, x*x*x]
    r = gdeco(l)
    return r

def mydeco2(c):
    def mkDeco(f):
        def inner(*args, **kw):
            args = [ [v*2 for v in args[0]], *args[1:] ]
            res = f(*args, **kw)
            return sqrt(abs(res*c))
        return inner
    return mkDeco

@mydeco2(1.23)
def gdeco2(l):
    return gl_sum(l)

def fdeco2(x):
    l = [x, x*x, x*x*x]
    r = gdeco2(l)
    return r

def mydeco3(f):
    def inner(*args, **kw):
        args = [ 2*args[0], *args[1:] ]
        res = f(*args, **kw)
        return sqrt(abs(res))
    return inner

@mydeco3
def fdeco3(x):
    return x*x

def mydeco4(c):
    def mkDeco(f):
        def inner(*args, **kw):
            args = [ c*args[0], *args[1:] ]
            res = f(*args, **kw)
            return sqrt(abs(res))
        return inner
    return mkDeco

@mydeco4(2.3)
def fdeco4(x):
    return x*x

def fdeco5(x):
    fd = mydeco3(f1)
    return fd(x)*x

def fdeco6(x):
    fd = mydeco4(2.75)(f1)
    return fd(x)*x

gfd1 = mydeco3(f1)
def fdeco7(x):
    return gfd1(x)*x

gfd2 = mydeco4(2.75)(f1)
def fdeco8(x):
    return gfd2(x)*x

glob_dict = {
    'd': {'a': 1, 'b': 2},
    'l': [1,2,3],
    'v': math.pi
}

glob_l = [1,2,3]

glob_t = [1,2,3],

def fglob(x):
    r1, r2, r3 = glist(*glob_l)
    return r1+r2+r3

def _flmult(x):
    l = [x] * 3
    l[1] = x*x
    l[2] = x*x*x
    z = gl_sum(x)
    return z

def _flitemassign(x):
    l = [1, 1, 1]
    l[0] += x
    l[1] -= x
    l[2] *= x
    z = gl_sum(l)
    return z

def faugass(x):
    if x < 0:
        x = -x
    y = x*x
    l = list([x,x,x,x,x,x,x,x])
    l[0] += y
    l[1] -= y
    l[2] *= y
#    l[3] @= y
    l[4] /= y
    l[5] //= y
    l[6] %= y
    l[7] **= y
    z = gl_sum(l)
    return z

def faugass2(x):
    if x < 0:
        x = -x
    l = list([x,x,x,x,x,x,x,x])
    l[0] += 2.5
    l[1] -= 2.5
    l[2] *= 2.5
#    l[3] @= 2.5
    l[4] /= 2.5
    l[5] //= 2.5
    l[6] %= 2.5
    l[7] **= 2.5
    z = gl_sum(l)
    return z


def mkcached():
    cache = {}

    def cached(f):

        def inner(x):
            nonlocal cache

            if x in cache:
                y = cache[x]
                # print(f'Return f({x}) from cache: {y}')
            elif isinstance(x, str):
                if x == 'clear':
                    cache = {}
                else:
                    raise ValueError()
                return 0
            else:
                y = f(x)
                cache[x] = y
                # print(f'Compute f({x}): {y}')
            return y

        return inner
    return cached


def mkcached2():

    def cached(f):
        cache = {}

        def inner(x):
            if x in cache:
                y = cache[x]
                # print(f'Return f({x}) from cache: {y}')
            else:
                y = f(x)
                cache[x] = y
                # print(f'Compute f({x}): {y}')
            return y

        return inner
    return cached


@mkcached()
def gcachedsin(x):
    return fsin(x)

def guncachedsin(x):
    return fsin(x)

def fcached(x):
    gcachedsin('clear')
    l = [x, x*x, x*x*x]
    v1 = [gcachedsin(v) for v in l]
    v2 = [gcachedsin(v) for v in l]
    return gl_sum(v1 + v2)

def fcached2(x):
    gcached = mkcached()(guncachedsin)
    l = [x, x*x, x*x*x]
    v1 = [gcached(v) for v in l]
    v2 = [gcached(v) for v in l]
    return gl_sum(v1 + v2)

def fcached3(x):
    gcached = mkcached2()(guncachedsin)
    l = [x, x*x, x*x*x]
    v1 = [gcached(v) for v in l]
    v2 = [gcached(v) for v in l]
    return gl_sum(v1 + v2)

gldata = {}
def gcachegl(x):
    if x in gldata:
        y = gldata[x]
    else:
        y = fsin(x)
        gldata[x] = y
    return y

def badfcachegl(x):
    l = [x, x*x, x*x*x]
    v1 = [gcachegl(v) for v in l]
    v2 = [gcachegl(v) for v in l]
    return gl_sum(v1 + v2)

def fwrglobal(x):
    l = [x, x*x, x*x*x]
    r, gldata['c'], _ = gtpl(*l)
    return r*2

def ggenerator(l):
    for i in range(len(l)):
        yield l[i]

def fgenerator(x):
    l = [x, x*x, x*x*x]
    vl = [v for v in ggenerator(l)]
    # print(f'Generated list {vl}')
    return gl_sum2(vl)

def md_ggenerator(dm_l, l):
    for (dm_i, i) in zip([0]*len(l), range(len(l))):
        yield (dm_l[i], l[i])

def fgeneratorm(x):
    l = [x, x*x, x*x*x]
    dl = [1, 0, 0]
    s = 0
    for (dm_v, v) in md_ggenerator(dl, l):
        s += v
    return s

def ggenerator2(l):
    for i in range(len(l)):
        if i == 0:
            yield l[i]
        elif i % 2 == 1:
            yield [sin(l[i]), cos(l[i])]
        else:
            yield fsin(l[i])*l[i-1] + l[i-2]


def fgenerator2(x):
    l = [x, x*x, x*x*x]
    l = l + l + l
    vl = [v for v in ggenerator2(l)]
    # print(f'Generated list {vl}')
    return gl_sum2(vl)
