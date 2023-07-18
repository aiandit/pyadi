import sys
from .astvisitor import isbuiltin
from .runtime import unjnd, dzeros


me = sys.modules[__name__]


call1 = [
    'binop_add', 'binop_sub', 'binop_mult', 'binop_matmult', 'binop_div', 'binop_floordiv', 'binop_mod', 'binop_pow',
    'unaryop_uadd', 'unaryop_usub'
]
call2 = ['super']


def mkCall(f):
    def run(*args, **kw):
        d_kw, kw = unjnd(kw)
        # print(f'Run function {f.__name__} ({args}), kw={kw}, d_kw={d_kw}')
        dres = f(*args[1::2], **kw)
        dres = dzeros(dres)
        res = f(*args[1::2], **kw)
        # print(f'dzeros {res} {dres}')
        return dres, res
    return run


def mkCall2(f):
    def run2(*args, **kw):
        d_kw, kw = unjnd(kw)
        # print(f'Run function {f.__name__} ({args}), kw={kw}, d_kw={d_kw}')
        dres = f(*args[0::2], **d_kw)
        res = f(*args[1::2], **kw)
        #dres = dzeros(res)
        #print(f'dzeros {res} {dres}')
        return dres, res
    return run2


def decorator(**opts):

    def inner(done, key, f, *args, **kw):

        # print(f'D1 {f.__qualname__} before')

        if f.__name__ in call2:
            return mkCall2(f)

        if isbuiltin(f) or f.__qualname__ in call1:
            return mkCall(f)

        res = done(key)

        # print(f'D1 {f.__qualname__} after')

        return res

    return inner

# (c) 2023 AI & IT UG
# Author: Johannes Willkomm jwillkomm@ai-and-it.de
