from itertools import chain
from math import sin, cos, tan, asin, acos, atan, log, sqrt, floor
from .astvisitor import getmodule, rid
from .runtime import unjnd
import sys
import numpy as np


me = sys.modules[__name__]


def mkRule(f, rule):
    def runRule(*args, **kw):
        d_kw, kw = unjnd(kw)
        res = f(*args[1::2], **kw)
        dres = rule(res, *args, **d_kw)
        return dres, res
    runRule.builtin = True
    return runRule

def mkRule2(f, rule):
    def runRule2(*args, **kw):
        return rule(*args, **kw)
    runRule2.builtin = True
    return runRule2

def initType(function, *args, **kw):
    do, o = function(*args[1::2], **kw), function(*args[1::2], **kw)
    do = dzeros(do)
    return do, o

def decorator(**opts):

    def inner(done, key, f):

        id = 'D_' + rid(f)
        rule = getattr(me, id, None)

        if rule is not None:
            return mkRule(f, rule)

        id = 'E_' + rid(f)
        rule = getattr(me, id, None)

        if rule is not None:
            return mkRule2(f, rule)

        # try source diff
        return done(key)

    return inner


dict = {}
hidden = {}


def D_pyfad_runtime_binop_add(r, dx, x, dy, y):       return dx+dy
def D_pyfad_runtime_binop_sub(r, dx, x, dy, y):       return dx-dy
def D_pyfad_runtime_binop_mult(r, dx, x, dy, y):      return dx*y + x*dy
def D_pyfad_runtime_binop_matmult(r, dx, x, dy, y):   return dx@y + x@dy
def D_pyfad_runtime_binop_div(r, dx, x, dy, y):       return (dx*y - x*dy) / y**2
def D_pyfad_runtime_binop_floordiv(r, dx, x, dy, y):  return 0
def D_pyfad_runtime_binop_mod(r, dx, x, dy, y):       return dx - dy * int(floor(x/y))
def D_pyfad_runtime_binop_pow(r, dx, x, dy, y):       return r * (dx * y / x + dy*log(x))

def D_pyfad_runtime_unaryop_uadd(r, dx, x): return +dx
def D_pyfad_runtime_unaryop_usub(r, dx, x): return -dx

def D_builtins_print(r, *args):
    print('D ', *args[0::2])
    return 0

def D_builtins_super(r, *args):
    print('D_super', *args)
    return super(*args[0::2])

def E_builtins_super___init__(*args, **kw):
    print('E_super___init___builtins', *args)
    return args[0], args[1]

def E_builtins_object___init__(*args, **kw):
    print('E_object___init___builtins', *args)
    return args[0], args[1]

def D_builtins_dict_items(r, dx, x):
    return dx.items()

def D_builtins_dict_keys(r, dx, x):
    return r

def D_builtins_dict_values(r, dx, x):
    return dx.values()

def D_builtins_range(r, *args):
    return [0]*len(r)

def D_builtins_len(r, dx, x):
    return 0

def D_builtins_enumerate(r, dx, x):
    return zip([0]*len(x), dx)

def D_builtins_list(r, dx, x):
    return list(dx)

def D_builtins_int(r, dx, x): return 0
def D_builtins_float(r, dx, x): return float(dx)
def D_builtins_complex(r, dx, x, dy, y): return complex(dx, dy)

def D_builtins_getattr(r, *args):
    return getattr(*args[0::2])

def D_builtins_Random_random(r):
    return 0

def D_builtins_min(r, dx, x, dy, y):
    return dx if x < y else dy

def D_builtins_max(r, dx, x, dy, y):
    return dx if not(x < y) else dy


def D_builtins_abs(r, dx, x):
    return 1 if not(x < 0) else -1


def D_builtins_sum(r, dx, x):
    return sum(dx)


def D_math_sin(r, dx, x):
    return dx * cos(x)

def D_math_cos(r, dx, x):
    return dx * -sin(x)

def D_math_tan(r, dx, x):
    return dx / cos(x)**2


def D_math_asin(r, dx, x):
    return dx / sqrt(1 - x**2)

def D_math_acos(r, dx, x):
    return -dx / sqrt(1 - x**2)

def D_math_atan(r, dx, x):
    return dx / (1 + x**2)


def D_math_log(r, dx, x):
    return dx / x

def D_math_sqrt(r, dx, x):
    return 0.5 * dx / r


def D_numpy_zeros(r, dshape, shape):
    return r.copy()

def D_numpy_savetxt(r, *args):
    return r

def D_numpy_matmul(r, dx, x, dy, y):
    return np.matmul(dx, y) + np.matmul(x, dy)

def E_builtins_ndarray_copy(*args, **kw):
    return args[0].copy(),  args[1].copy()

def D_numpy_sum(r, dx, x):
    return np.sum(dx)
