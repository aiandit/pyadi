from itertools import chain
from astunparse.astnode import isgeneric, fields

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
    print('unzd', d)
    keys = d.keys()
    dvals, vals = zip(*d.values())
    d_r, r = dict(zip(keys, dvals)), dict(zip(keys, vals))
    return d_r, r
