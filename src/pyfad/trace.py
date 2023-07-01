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

class Stop(BaseException):
    pass

def decorator(**opts):

    print('Create decorator: ', opts.items())

    tracecalls = opts.get('tracecalls', False)
    timeout = opts.get('timeout', None)

    data = {}
    x = 17

    if tracecalls:
        data['hist'] = []
        print(f'create hist in {data}')

    if timeout is not None:
        stop_t0 = time.time()


    def inner(done, key, f, *args, **kw):

        print(f'call to {f.__name__}{(*args,)}) starts {time.time()} s')

        if tracecalls:
            data['hist'] += [f.__qualname__]
            print(f'add hist {data["hist"]} {[f.__qualname__]}')

        if timeout is not None:
            dt = time.time() - stop_t0
            if dt > timeout:
                raise Stop(f'Stopping because {dt} s have passed.')

        if 'stop' in data:
            raise Stop(f'Stopping because stop set.')

        res = done(key)

        print(f'call to {f.__name__}{(*args,)}) = {res} ends {time.time()} s')

        return res

    def get(*args, **kw):
        for name in args:
            del data[name]
        data.update(kw)
        if kw.get('get'):
            return data[kw.get('get')]
        return data

    return inner, get
