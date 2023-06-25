from math import sin, cos, tan, asin, acos, atan, log, sqrt

def f1(x,y,z):
    r = x*2*x*y*z
    return r

def f2(x,y,z):
    r = x*y*z*17
    return r

def f3(x,y,z):
    r = x*f1(y*z*17,y*z,z)
    return r

def f4(x,y,z):
    s = f2(z,y,x)
    r = x*f1(y*z*17,y*z,z)*f3(s, y, x)
    return r

def f5(x,y,z):
    r = x*sin(y)
    return r

def f6(x,y,z):
    a = 17
    b = 2.3 * a
    r = x*y*z*a*b
    return r

def f7(x,y,z):
    a = 17
    b = 2.3 * a
    r = f1(x*y,z*y,a*b)
    return r

def f8(x,y,z):
    a = 17
    b = 2.3 * a
    r = f6(z=x*y,y=z*y,x=a*b)
    return r

def f9(x,y,z):
    a = 17
    b = 2.3 * a
    r = f6(a*b,z=x*y,y=z*y)
    return r

def fdef(x,y,z=2):
    a = 17
    b = 2.3 * a
    r = x*y*z*a*b
    return r

def f10(x,y,z):
    a = 17
    b = 2.3 * a
    r = fdef(y=a*b,x=x*y)
    return r

gx = 1.23

#TODO: globals
def _fdef2(x,y,z=gx*2):
    a = 17
    b = 2.3 * a
    r = x*y*z*a*b
    return r

def _f11(x,y,z):
    a = 17
    b = 2.3 * a
    r = fdef2(y=a*b,x=x*y)
    return r

def f12(x,y,z):
    a = 17
    b = 2.3 * a
    l = [x, y*2, z*y]
    s = l[-1] * x
    r = f1(l[0], x*y, s)
    return r
