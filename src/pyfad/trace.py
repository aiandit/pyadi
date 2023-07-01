import time
import sys

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

    tracecalls = opts.get('tracecalls', False)
    verbose = opts.get('verbose', False)

    data = {}

    if tracecalls:
        hist = []


    def inner(done, key, f, *args, **kw):
        nonlocal hist

        if verbose:
            print(f'call to {f.__name__}{(*args,)}) starts {time.time()} s')

        if tracecalls:
            data['cur'] = f.__qualname__
            hist += [f.__qualname__]

        if 'cmd' in data:
            while True:
                cmd = data.get('cmd', '')
                print(f'Command set: {cmd}')
                if cmd == 'pause':
                    print(f'condition wait on {data["condition"]}')
                    sys.stdout.flush()
                    data['condition'].wait()
                    print('been notified')
                    sys.stdout.flush()
                elif cmd == 'sleep':
                    time.sleep(data['timeout'])
                else:
                    if cmd == 'stop':
                        raise Stop(f'Stopping because stop set: {data["msg"]}.')
                    elif cmd == 'raise':
                        raise data['exception']
                    else:
                        if cmd == 'call':
                            data['function'](data, hist, f, args, kw)
                    break

        res = done(key)

        if verbose:
            print(f'call to {f.__name__}{(*args,)}) = {res} ends {time.time()} s')

        return res

    def get(*args, **kw):
        nonlocal hist
        for name in args:
            del data[name]
        if kw.get('get') == 'hist':
            return hist
        elif kw.get('clear') == 'hist':
            hist = []
        elif kw.get('get'):
            return data[kw.get('get')]
        else:
            data.update(kw)
        return data

    return inner, get
