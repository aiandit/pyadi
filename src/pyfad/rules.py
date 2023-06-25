from math import sin, cos, tan, asin, acos, atan, log, sqrt

def D_sin_math(dx, x):
    return dx * cos(x), sin(x)

def D_cos_math(dx, x):
    return dx * -sin(x), cos(x)

def D_sqrt_math(dx, x):
    r = sqrt(x)
    return 0.5 * dx / r, r
