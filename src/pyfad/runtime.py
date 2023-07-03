from astunparse.astnode import isgeneric, fields

def dzeros(args):
    #print(f'dzeros {args}')
    if isinstance(args, list):
        return [dzeros(f) for f in args]
    elif isinstance(args, tuple):
        return tuple([dzeros(f) for f in args])
    elif isinstance(args, dict):
        return {f: dzeros(v) for f, v in args.items()}
    elif isinstance(args, float) or isinstance(args, complex) or isinstance(args, int):
        return 0.0
    elif isinstance(args, str) or isinstance(args, bytes) or isinstance(args, bytearray):
        return args
    elif hasattr(args, '__iter__'):
        return [dzeros(v) for v in args]
    elif isinstance(args, object):
        # we assume the object is already allocated
        try:
            for a in fields(args, True):
                setattr(args, a, dzeros(getattr(args, a)))
        except BaseException as ex:
            print(ex)
            pass
        return args
    return args


def unjnd(d):
    names = list(d.keys())
    values = list(d.values())
    dd = dict(zip(names[len(names)//2:], values[0:len(names)//2]))
    d = dict(zip(names[len(names)//2:], values[len(names)//2:]))
    return dd, d


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
