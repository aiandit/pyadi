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
