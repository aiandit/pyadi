from .astvisitor import getmodule

def isbuiltin(func):
    mod, modfile = getmodule(func)
    res = modfile is None
#    print('isbuiltin', func, res)
    return res

def decorator(done):

    def inner(f, dargs, args, **kw):
        res = done()

        if isbuiltin(f):
            return res, res
        else:
            return None, res

    return inner

