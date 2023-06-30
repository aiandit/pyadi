from math import sin, cos, tan, asin, acos, atan, log, sqrt

import math

def f1(x):
    z = fsin(x)
    return z

def f2(x):
    z = 2*x
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
    return z

def fmod2(x):
    r = -sin(x)
    s = r % 3
    return s

def fpow(x):
    if x < 0:
        x = -x
    r = sin(x)
    s = r ** x
    t = s ** 2
    z = 2 ** t
    return z

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

def _fobj(x):
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

def _fplane(x):
    o = Plane()
    o.fly(x)
    r = o.distance + o.gas
    return r

class Plane2:
    distance = 0
    velocity = 100
    gas = 1e4
    consumption = 10
    def __init__(self, c):
        self.consumption = c
    def fly(self, t):
        dist = self.velocity * t
        self.distance += dist
        self.gas -= self.consumption * dist

def _fplane2(x):
    y = x*x
    o = Plane2(x)
    o.fly(y)
    r = o.distance + o.gas
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

def fprint(x):
    l = [f1(x), 2*x, 3*x]
    print(l)
    assert len(l) == 3
    if x == 0:
        raise(ValueError())
    return gl_sum2(l)

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

def fcalll4(x):
    l = [x, x*x, x*x*x]
    A = gdiag(l)
    v = gdiag(A)
    assert gl_sum2(A) == gl_sum2(v)
    assert gl_sum2(A) == gl_sum2(l)
    z = g2l(gl_sum2(A), v)
    return z
