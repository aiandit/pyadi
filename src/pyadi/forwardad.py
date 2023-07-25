"""The default rule module for PyADi.

It resolves functions for which it has a handler registered to a
corresponding function compatible with our AD process, that is,
function that takes the flattened list of 2*N arguments and the
differentiated keywords dictionary and returns a tuple of the
derivative and the function result.

When some function is not handled by this module, then source
transformation will be invoked. This in turn fails with the exception
:py:exc:`.NoSource` when the source can not be obtained. Then a
handler must be added to this module in order to make the process
work.

Defining handlers, how they are resolved and their semantics is
entirely local to this module.

Handlers for any Python function ``function`` can be set, retrieved,
and deleted with :py:func:`.setrule`, :py:func:`.getrule`, and
:py:func:`.delrule`, respectively.

Note the mode parameter of these functions, which can be 'D', 'Dkw' or
'E'. This controls how the generic calling pattern is mapped to the
rules:

  "D") :py:func:`.mkRule` is used to build the AD function, and the
       rule must have the signature ``rule(r, *args, **kw)``, or
       compatible. The wrapper calls the original function first as
       ``r = function(*args[1::2])`` and then calls the rule passing
       the result ``r`` as the first parameter. The rule must return
       just the derivative ``d_r``, the wrapper returns ``(d_r, r)``.

  "Dkw") similar to "D", :py:func:`.mkRule2` is used to build the AD
         function, which additionally unpacks the keywords into the
         derivative keywords and the original keywords with
         :py:func:`.unjnd`. The rule must have the signature ``rule(r,
         d_kw, *args, **kw)``. The wrapper in addition to the function
         result passes the derivative keywords as the second parameter
         ``d_kw`` while ``kw`` will only receive the original keyword
         params.

  "E") No wrapper is produced, the rule will be called directly, which
       accordingy must return a tuple.

Note that the rule modes have the precedence 'D', 'Dkw', and 'E'. So
if you wanted to replace an existing rule with mode 'D' with a new
rule with mode 'E', make sure to :py:func:`.delrule` the handler with
mode 'D' first, or else your new rule would be shadowed by the still
extant rule with mode 'D'.

Mode "D" is useful in that vast majority of cases. For example, the
handler for :py:func:`~math.sin` is::

  def D_math_sin(r, dx, x):
      return dx * cos(x)

The handler for :py:func:`print` calls print again, this time with all
the differentiated arguments::

  def D_builtins_print(r, *args):
      print('D ', *args[0::2])
      return 0

The function result that is provided by the wrapper can be put to good
use in several functions, for example the exponentials like
:py:func:`~math.exp` and :py:func:`~math.sqrt`, but also
:py:func:`numpy.linalg.inv`::

  def D_math_exp(r, dx, x):
      return r * dx

  def D_math_sqrt(r, dx, x):
      return 0.5 * dx / r

  def D_numpy_linalg_inv(r, dx, x):
      return r @ -dx @ r

Common trivial cases are type conversions, which should be applied to
the derivative arguments too::

  def D_builtins_list(r, dx, x):
      return list(dx)

  def D_builtins_tuple(r, dx, x):
      return tuple(dx)

  def D_builtins_float(r, dx, x):
      return float(dx)

  def D_builtins_complex(r, dx, x, dy, y):
      return complex(dx, dy)

Another generic case is when data is shuffled around in whatever way,
then apply the same shuffling to the derivatives::

  def D_builtins_ndarray_reshape(r, *args, **kw):
      # args[0]: the derivative array
      # args[1]: the original array
      # args[3::2]: the remainder of the original arguments, i.e. the sizes
      return args[0].reshape(*args[3::2])

  def D_numpy_hstack(r, dx, x):
      return np.hstack(dx)

  def D_numpy_diag(r, dx, x):
      return np.diag(dx)

  def D_numpy_real(r, dx, x):
      return np.real(dx)

  def D_numpy_imag(r, dx, x):
      return np.imag(dx)


Another trivial case, but which can be a pitfall, is where values are
created that do not depend on the inputs, in which case it is
paramount that the derivative values are always zero::

  def D_builtins_int(r, dx, x):
      return 0

  def D_builtins_len(r, dx, x):
      return 0

  def D_builtins_Random_random(r):
      return 0

When a sequence is created, we must provide a sequence of zeros::

  def D_builtins_range(r, *args):
      return [0]*len(r)

  def D_builtins_enumerate(r, dx, x):
      return zip([0]*len(x), dx)

And likewise for arrays::

  def D_numpy_zeros(r, *args):
      return np.zeros(r.shape, dtype=r.dtype)

  D_numpy_eye = D_numpy_ones = D_numpy_random_rand = D_numpy_zeros

The intention of the mode "Dkw" is to save the call to
:py:func:`.unjnd` when it is not needed. The flipside is that rules
must announce when they do want the split keyword arguments, but so
far these are quite a small number, for example, :py:func:`dict`
itself::

  def Dkw_builtins_dict(r, d_kw, *args, **kw):
      return dict(**d_kw)

Another example is :py:func:`numpy.sum`, which on past occasions has
it made pretty clear that it does not want to be bothered with a
keyword ``d_axis`` or any other keyword starting with ``d_`` for that
matter, so we give it what it needs but not what is does not want::

  def Dkw_numpy_sum(r, d_kw, dx, x, **kw):
      return np.sum(dx, **kw)

"""

from itertools import chain
from math import sin, cos, tan, asin, acos, atan, log, sqrt, floor
from .astvisitor import getmodule, rid
from .runtime import unjnd
import sys
import numpy as np


me = sys.modules[__name__]


def getrule(func, adfunc, mode='D'):
    id = mode + '_' + rid(func)
    return getattr(me, id)


def setrule(func, adfunc, mode='D'):
    id = mode + '_' + rid(func)
    setattr(me, id, adfunc)


def delrule(func, mode='D'):
    id = mode + '_' + rid(func)
    res = getattr(me, id)
    delattr(me, id)
    return res


def getrules():
    return me.__dict__


def mkRule(f, rule):
    def runRule(*args, **kw):
        res = f(*args[1::2], **kw)
        dres = rule(res, *args, **kw)
        return dres, res
    return runRule


def mkRule2(f, rule):
    def runRule2(*args, **kw):
        d_kw, kw = unjnd(kw)
        res = f(*args[1::2], **kw)
        dres = rule(res, d_kw, *args, **kw)
        return dres, res
    return runRule2


def mkRule3(f, rule):
    return rule


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

        id = 'Dkw_' + rid(f)
        rule = getattr(me, id, None)

        if rule is not None:
            return mkRule2(f, rule)

        id = 'E_' + rid(f)
        rule = getattr(me, id, None)

        if rule is not None:
            return mkRule3(f, rule)

        # try source diff
        return done(key)

    return inner



def D_pyadi_runtime_binop_add(r, dx, x, dy, y):       return dx+dy
def D_pyadi_runtime_binop_sub(r, dx, x, dy, y):       return dx-dy
def D_pyadi_runtime_binop_mult(r, dx, x, dy, y):      return dx*y + x*dy
def D_pyadi_runtime_binop_c_mult(r, dx, x, dy, y):    return x*dy
def D_pyadi_runtime_binop_d_mult(r, dx, x, dy, y):    return dx*y
def D_pyadi_runtime_binop_matmult(r, dx, x, dy, y):   return dx@y + x@dy
def D_pyadi_runtime_binop_div(r, dx, x, dy, y):       return (dx*y - x*dy) / y**2
def D_pyadi_runtime_binop_floordiv(r, dx, x, dy, y):  return 0
def D_pyadi_runtime_binop_mod(r, dx, x, dy, y):       return dx - dy * int(floor(x/y))
def D_pyadi_runtime_binop_pow(r, dx, x, dy, y):       return dx * y * (x ** (y -1)) + ((dy * r * np.log(x)) if dy != 0 else 0)

def D_pyadi_runtime_unaryop_uadd(r, dx, x): return +dx
def D_pyadi_runtime_unaryop_usub(r, dx, x): return -dx

def D_builtins_print(r, *args):
    print('D ', *args[0::2])
    return 0

# super is given two superfluous excess arguments now...
def E_builtins_super(*args):
    return super(*args[2::2]), super(*args[3::2])

def E_builtins_super___init__(*args, **kw):
    #print('E_super___init___builtins', *args)
    return args[0], args[1]

def E_builtins_object___init__(*args, **kw):
    #print('E_object___init___builtins', *args)
    return args[0], args[1]

def Dkw_builtins_dict(r, d_kw, *args, **kw):
    return dict(**d_kw)

def D_builtins_dict_items(r, dx, x):
    return dx.items()

def D_builtins_dict_keys(r, dx, x):
    return r

def D_builtins_dict_values(r, dx, x):
    return dx.values()

def D_builtins_dict_get(r, dx, x, dk, k, dd, d):
    return dx.get(dk, dd)

def D_builtins_list(r, dx, x):
    return list(dx)

def D_builtins_tuple(r, dx, x):
    return tuple(dx)

def D_builtins_range(r, *args):
    return [0]*len(r)

def D_builtins_len(r, dx, x):
    return 0

def D_builtins_enumerate(r, dx, x):
    return zip([0]*len(x), dx)

def D_builtins_zip(r, *args):
    return zip(*args[0::2])

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
    return dx * (1 if not(x < 0) else -1)


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


def D_math_exp(r, dx, x):
    return r * dx

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

def E_builtins_ndarray_reshape(*args, **kw):
    return args[0].reshape(*args[3::2]), args[1].reshape(*args[3::2])

def D_numpy_hstack(r, dx, x):
    return np.hstack(dx)

def Dkw_numpy_sum(r, d_kw, dx, x, **kw):
    return np.sum(dx, **kw)

def D_numpy_diag(r, dx, x):
    return np.diag(dx)

def D_numpy_zeros(r, *args):
    return np.zeros(r.shape, dtype=r.dtype)

D_numpy_eye = D_numpy_ones = D_numpy_random_rand = D_builtins_RandomState_rand = D_numpy_zeros

def D_numpy_array(r, dx, x):
    return np.array(dx)

def D_numpy_linalg_inv(r, dx, x):
    return r @ -dx @ r

def D_numpy_linalg_norm(r, dx, x, d_ord=0, ord=None):
    if ord is None:
        return np.sum(x * dx) / r
    else:
        if ord == 2:
            if x.ndim > 1:
                U, S, Vh = np.linalg.svd(x)
                maxv = S[0]
                dn = U[:,0] @ dx @ Vh[0,:]
                return dn
            else:
                raise ValueError()
                return np.sum(x * dx) / r
        else:
            return np.sum(x * dx) / r

def D_numpy_real(r, dx, x):
    return np.real(dx)

def D_numpy_imag(r, dx, x):
    return np.imag(dx)

def D_numpy_exp(r, dx, x):
    return r * dx

def D_numpy_log(r, dx, x):
    return dx / x

def D_numpy_sqrt(r, dx, x):
    return 0.5 * dx / r

def D_numpy_sin(r, dx, x):
    return dx * np.cos(x)

def D_numpy_cos(r, dx, x):
    return dx * -np.sin(x)

def D_numpy_tan(r, dx, x):
    return dx / np.cos(x)**2

# (c) 2023 AI & IT UG
# Author: Johannes Willkomm jwillkomm@ai-and-it.de
