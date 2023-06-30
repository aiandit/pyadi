from .astvisitor import getmodule

def isbuiltin(func):
    mod, modfile = getmodule(func)
    res = modfile is None
#    print('isbuiltin', func, res)
    return res

def decorator(done):

    def inner(f, *args, **kw):

        print(f'D1 {f.__name__} before')

        res = done()

        assert res == None

        if isbuiltin(f):
            print('Call f')
            r = f(*args[1::2], **kw)
            res = r, r
        else:
            res = None, None

        print(f'D1 {f.__name__} after {res}')
        return res

    return inner
