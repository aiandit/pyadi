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
            print('Dummy call to builtin rule')
            res = f(*args[1::2], **kw)
            return res, res

        res = done(key)

        print(f'D2 {f.__name__} after')

        return res

    return inner
