import time

before = True
after = True
always = True

def select(mode, module, f):
    id = 'D_' + mode
    res = getattr(module, id, None)
    print(f'select.trace: {mode}, {module.__name__}, {f.__name__} => {id} {res is not None}')
    return res

def D_before(res, f, dargs, args, **kw):
    print(f'call to {f.__name__}{(*args,)}) starts {time.time()} s')

def D_after(res, f, dargs, args, **kw):
    print(f'call to {f.__name__}{(*args,)}) = {res} ends {time.time()} s')
