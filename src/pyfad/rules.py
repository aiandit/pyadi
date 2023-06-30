from itertools import chain

from .astvisitor import getmodule
from . import forwardad
from . import trace
from . import dummy, dummy2



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
