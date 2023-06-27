from math import sin, cos, tan, asin, acos, atan, log, sqrt

import math

def f1(x):
    z = fsin(x)
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
    z = sqrt(x)
    return z

def flog(x):
    z = log(x)
    return z

def fasin(x):
    z = asin(x)
    return z

def fatan(x):
    z = atan(x)
    return z

def fdiv(x):
    s = sin(x)
    z = s / x
    return z

def gbabylonian(x, y=1):
    if abs(y**2 - x) < 1e-7:
        return y
    else:
        r = gbabylonian(x, (y + x / y) / 2)
        return r

def fbabylonian(x):
    r = gbabylonian(x)
    return r

def gbabylonian2(x, y=1):
    if abs(y**2 - x) < 1e-7:
        return y
    else:
        return gbabylonian(x, (y + x / y) / 2)

def fbabylonian2(x):
    return gbabylonian2(x)

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

def flist(x):
    s = gl_sum([x, x*x, x*x*x])
    return s

def flist2(x):
    s = gl_sum([f1(x)])
    return s
