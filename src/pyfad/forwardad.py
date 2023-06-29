from itertools import chain
from math import sin, cos, tan, asin, acos, atan, log, sqrt

def select_():
    pass

def czip(a, b):
    return chain(*zip(a, b))

def callmap(r, f, dargs, args):
    return [r] + list(czip(dargs,args))

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

def D_list_builtins(r, dx, x):
    return list(dx)

def D_Random_random_builtins(r):
    return 0

def D_min_builtins(r, dx, x, dy, y):
    return dx if x < y else dy

def D_max_builtins(r, dx, x, dy, y):
    return dx if not(x < y) else dy


def D_abs_builtins(r, dx, x):
    return 1 if not(x < 0) else -1


def D_sum_builtins(r, dx, x):
    return sum(dx)


def D_sin_math(r, dx, x):
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
