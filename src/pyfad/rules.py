from math import sin, cos, tan, asin, acos, atan, log, sqrt


dict = {}
hidden = {}


def D_dict_items_builtins(r, dx, x):
    return dx.items()

def D_range_builtins(r, dx, x):
    return [0]*len(r)

def D_len_builtins(r, dx, x):
    return 0

def D_enumerate_builtins(r, dx, x):
    return zip([0]*len(x), dx)


def D_sin_math(r, dx, x):
    print('dsin', r, dx, x)
    return dx * cos(x)

def D_cos_math(r, dx, x):
    return dx * -sin(x)

def D_tan_math(r, dx, x):
    return dx / cos(x)**2


def D_asin_math(r, dx, x):
    return dx / sqrt(1 - x**2)

def D_acos_math(r, dx, x):
    return -dx / sqrt(1 - x**2)

def D_atan_math(r, dx, x):
    return dx / (1 + x**2)


def D_log_math(r, dx, x):
    return dx / x

def D_sqrt_math(r, dx, x):
    return 0.5 * dx / r
