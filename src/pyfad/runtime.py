from itertools import chain
from astunparse.astnode import isgeneric, fields
import numpy as np

def dzeros(args, lev=0):
    #print(f'dzeros {args}')
    lev += 1
    if isinstance(args, type) or lev > 3:
        return args
    elif isinstance(args, list):
        return [dzeros(f, lev) for f in args]
    elif isinstance(args, tuple) or args.__class__.__name__ in ['dict_values', 'dict_keys', 'dict_items']:
        return tuple([dzeros(f, lev) for f in args])
    elif isinstance(args, dict):
        return {f: dzeros(v, lev) for f, v in args.items()}
    elif hasattr(args, 'flat'):
        return np.zeros(args.shape)
    elif isinstance(args, int):
        return 0
    elif isinstance(args, float):
        return 0.0
    elif isinstance(args, complex):
        return complex(0.0)
    elif isinstance(args, str) or isinstance(args, bytes) or isinstance(args, bytearray):
        return args
    elif isinstance(args, object):
        fnames = list(chain(*map(lambda c: fields(c, True), args.__class__.__mro__)))
        try:
            for a in fnames:
                setattr(args, a, dzeros(getattr(args, a), lev))
        except BaseException as ex:
            print(ex)
            pass
        return args
    return args


def unjnd(d):
    if d:
        names = list(d.keys())
        values = list(d.values())
        dd = dict(zip(names[len(names)//2:], values[0:len(names)//2]))
        d = dict(zip(names[len(names)//2:], values[len(names)//2:]))
        return dd, d
    else:
        return {}, {}


def joind(ddl, dl):
    res = {}
    for dd in ddl:
        res |= { 'd_' + k: v for k, v in dd.items() }
    for d in dl:
        res |= { k: v for k, v in d.items() }
    return res


def unzd(d):
    # print('unzd', d)
    if d:
        keys = d.keys()
        dvals, vals = zip(*d.values())
        d_r, r = dict(zip(keys, dvals)), dict(zip(keys, vals))
        return d_r, r
    else:
        return {}, {}

class DWith:
    def __init__(self, *args, **kw):
        self.dobj, self.obj = args[0]
        super().__init__(*kw)

    def __enter__(self, *args, **kw):
        self.dobj.__enter__(*args, **kw)
        self.obj.__enter__(*args, **kw)
        return self.dobj, self.obj

    def __exit__(self, *args, **kw):
        self.dobj.__exit__(*args, **kw)
        self.obj.__exit__(*args, **kw)


def binop_add(x, y): return x+y
def binop_sub(x, y): return x-y
def binop_mult(x, y): return x*y
def binop_c_mult(x, y): return x*y
def binop_d_mult(x, y): return x*y
def binop_matmult(x, y): return x@y
def binop_div(x, y): return x/y
def binop_floordiv(x, y): return x//y
def binop_mod(x, y): return x%y
def binop_pow(x, y): return x**y

def unaryop_uadd(x): return +x
def unaryop_usub(x): return -x

def augassign_add(x, y): return x+y
def augassign_sub(x, y): return x-y
def augassign_mult(x, y): return x*y
def augassign_div(x, y): return x/y
def augassign_truediv(x, y): return x//y
def augassign_mod(x, y): return x%y
