from itertools import chain

from .astvisitor import getmodule
from . import forwardad
from . import trace
from . import dummy, dummy2


class NoRule(BaseException):
    pass

rulemodules = {}

def clearrulemodules(name=None):
    global rulemodules
    rulemodules = {}

def addrulemodule(module, **kw):
    rulemodules[module.__file__] = module

def initRules(rules='ad'):
    clearrulemodules()
    rules = rules.split(',')
    for i in rules:
        if i == 'trace':
            addrulemodule(trace)
        elif i == 'ad':
            addrulemodule(forwardad)
        elif i == 'dummy':
            addrulemodule(dummy)
        elif i == 'dummy2':
            addrulemodule(dummy2)

def processRules(function, args, kw):
    state = [0]
    mkeys = list(rulemodules.keys())

    def nextStep():
        if state[0] >= len(mkeys):
            lenkw = len(kw)
#            print('kw', kw)
            return None
        else:
            ind = state[0]
            print(f'process {function.__name__} ', ind, mkeys[ind])
            print('args', args)
            assert len(args) == 0 or all([len(list(a)) == 2 for a in list(args)])
            state[0] += 1

            deco = rulemodules[mkeys[ind]].decorator(nextStep)

            dres = deco(function, *args, **kw)
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
