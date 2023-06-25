from math import sin, cos, tan, asin, acos, atan, log, sqrt

def fsin(x):
    z = sin(x)
    return z

def fcos(x):
    z = cos(x)
    return z

def ftan(x):
    z = tan(x)
    return z

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
    s = sqrt(x)
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
