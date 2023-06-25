from math import sin, cos, tan, asin, acos, atan, log, sqrt

dict = {}
hidden = {}

def D_dict_items_builtins(dx, x):
    return dx.items(), x.items()


def D_sin_math(dx, x):
    return dx * cos(x), sin(x)

def D_cos_math(dx, x):
    return dx * -sin(x), cos(x)

def D_tan_math(dx, x):
    return dx / cos(x)**2, cos(x)


def D_asin_math(dx, x):
    return dx / sqrt(1 - x**2), asin(x)

def D_acos_math(dx, x):
    return -dx / sqrt(1 - x**2), acos(x)

def D_atan_math(dx, x):
    return dx / (1 + x**2), atan(x)


def D_log_math(dx, x):
    return dx / x, log(x)

def D_sqrt_math(dx, x):
    r = sqrt(x)
    return 0.5 * dx / r, r
