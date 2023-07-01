from .astvisitor import getmodule

def isbuiltin(func):
    mod, modfile = getmodule(func)
    res = modfile is None
#    print('isbuiltin', func, res)
    return res

def decorator(**opts):

    def inner(done, key, f, *args, **kw):

        print(f'D1 {f.__name__} before')

        if isbuiltin(f):
            print('Call f')
            r = f(*args[1::2], **kw)
            return 0, r

        res = done(key)

        print(f'D2 {f.__name__} after')

        return res

    return inner
