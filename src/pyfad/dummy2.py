from .astvisitor import getmodule

def isbuiltin(func):
    mod, modfile = getmodule(func)
    res = modfile is None
#    print('isbuiltin', func, res)
    return res

def decorator(done):

    def inner(f, dargs, args, **kw):

        print(f'D2 {f.__name__} before')

        res = done()

        print(f'D2 {f.__name__} after {res}')

        return res

    return inner
