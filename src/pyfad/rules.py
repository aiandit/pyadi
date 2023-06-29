from itertools import chain

from .astvisitor import getmodule
from . import forwardad
from . import trace

def czip(a, b):
    return chain(*zip(a, b))

def callmapDefault(res, f, dargs, args, **kw):
    return (res, f, dargs, args)

def callmapAD(rule, f, dargs, args, **kw):
    return rule(f, czip(dargs, args), **kw)

def selectDefault(mode, module, f):
    id = ('C_' if mode == 'before' else 'D_') + rid(f)
    res = getattr(module, id, None)
#    print(f' **** select: {mode}, {module.__name__}, {f.__name__} => {id} {res}')
    return res

def selectMode(module, f, mode):
    id = 'D_' + mode
    res = getattr(module, id, None)
#    print(f'select: {mode}, {module.__name__}, {f.__name__} => {id} {res}')
    return res

class NoRule(BaseException):
    pass

class RuleModule:

    def __init__(self, module, before=None, after=None, callmap=None, silent=False, select=None, replace=False):
        self.module = module
        self.callmap = callmap if callmap is not None else getattr(module, 'callmap', callmapDefault)
        self.select = select if select is not None else getattr(module, 'select', selectDefault)
        self.before = before if before is not None else getattr(module, 'before', False)
        self.after = after if after is not None else getattr(module, 'after', True)
        self.silent = silent
        self.replace = replace if replace is not None else getattr(module, 'replace', False)
#        print('RM init', vars(self))

    def call(self, done, f, dargs, args, **kw):
#        print(f'call: {self.module.__name__}, {f.__name__}')
        if self.before:
#            print(f'call.before: {self.module.__name__}, {f.__name__}')
            bfun = self.select('before', self.module, f)
            if bfun:
                bres = bfun(*self.callmap(None, f, dargs, args), **kw)
#        print(f'call.function: {self.module.__name__}, {f.__name__}')
        (dres, res) = done()
#        print(f'call.function: {self.module.__name__}, {f.__name__} res: {dres, res}')
        if self.replace:
            res = bres
        if self.after:
#            print(f'check.after: {self.module.__name__}, {f.__name__} {self.select.__module__}.{self.select.__qualname__}')
            afun = self.select('after', self.module, f)
            if afun:
#                print(f'call.after: {self.module.__name__}, {f.__name__} => {afun.__name__}')
                ares = afun(*self.callmap(res, f, dargs, args), **kw)
#                print(f'call.dres: {ares}')
                if ares is not None:
                    dres = ares
#        print(f'return from: {self.module.__name__}, {f.__name__} {(dres, res)}')
        return (dres, res)

rulemodules = {}

def clearrulemodules(name=None):
    global rulemodules
    rulemodules = {}

def addrulemodule(module, **kw):
    r = RuleModule(module, **kw)
    rulemodules[module.__file__] = r

def initRules():
#    addrulemodule(trace)
    addrulemodule(forwardad)

def processRules(function, *args, **kw):
    state = [0]
    mkeys = list(rulemodules.keys())
    def nextStep():
        if state[0] >= len(mkeys):
            lenkw = len(kw)
#            print('kw', kw)
            return (None, function(*args[1], **{f: kw[f] for i, f in enumerate(kw) if i >= lenkw/2 }))
        else:
            ind = state[0]
#            print('process ', ind, mkeys[ind])
#            print('args', args)

            state[0] += 1
            dres = rulemodules[mkeys[ind]].call(nextStep, function, *args, **kw)
#            print('process rules ', ind, 'res', dres)

            return dres
    return nextStep()



def rid(func):
    mod, _ = getmodule(func)
    fid = f'{func.__qualname__}_{mod}'.replace('.', '_')
#    print('Rule ID', func, fid)
    return fid


def setrule(func, adfunc):
    id = 'D_' + rid(func)
    print(f'set AD rule for {func.__name__}, key {id}')
    setattr(forwardad, id, adfunc)
    forwardad.dict[id] = adfunc


def delrule(func):
    id = 'D_' + rid(func)
    print(f'clear AD rule for {func.__name__}, key {id}')
    if id in forwardad.dict:
        del forwardad.dict[id]
    else:
        forwardad.hidden[id] = getattr(forwardad, id)
    delattr(forwardad, id)


def restorerule(func):
    id = 'D_' + rid(func)
    print(f'restore AD rule for {func.__name__}, key {id}')
    if id in forwardad.hidden:
        setattr(forwardad, id, forwardad.hidden[id])
        del forwardad.hidden[id]


def getrules():
    return forwardad.dict
