from .astvisitor import isbuiltin
from .timer import Timer

def decorator(**opts):

    catch = opts.get('catch', [])
    height = opts.get('height', 1)
    stack = 0
    found = 0

    def inner(done, key, f):

        adfun = done(key)

        def timing(*args, **kw):
            nonlocal found, stack
            stack += 1
            if f.__name__ in catch:
                found = stack

            if stack < found + height:
                with Timer(f.__qualname__, f'time-{adfun.__name__}-{found}-{stack}') as t:
                    res = adfun(*args, **kw)
            else:
                res = adfun(*args, **kw)

            stack -= 1
            return res

        return timing

    return inner
