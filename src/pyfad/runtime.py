from astunparse.astnode import isgeneric, fields

def dzeros(args):
    if isinstance(args, list):
        return [dzeros(f) for f in args]
    elif isinstance(args, tuple):
        return tuple([dzeros(f) for f in args])
    elif isinstance(args, dict):
        return {f: dzeros(v) for f, v in args.items()}
    elif isgeneric(args):
        return 0.0
    elif isinstance(args, object):
        # we assume the object is already allocated
        for a in fields(args, True):
            setattr(args, a, dzeros(getattr(args, a)))
        return args
    return args

