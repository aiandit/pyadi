from .astvisitor import isbuiltin

def mkCall(f):
    def run(*args, **kw):
        res = f(*args[1::2], **kw)
        return res, res
    return run

def decorator(**opts):

    def inner(done, key, f, *args, **kw):

        print(f'D1 {f.__name__} before')

        if isbuiltin(f):
            return mkCall(f)

        res = done(key)

        print(f'D2 {f.__name__} after')

        return res

    return inner
