from itertools import chain

from .astvisitor import getmodule
from . import forwardad


def rid(func):
    mod, _ = getmodule(func)
    fid = f'{func.__qualname__}_{mod}'.replace('.', '_')
#    print('Rule ID', func, fid)
    return fid


def getrule(func, adfunc):
    id = 'D_' + rid(func)
    return getattr(forwardad, id)


def setrule(func, adfunc):
    id = 'D_' + rid(func)
    setattr(forwardad, id, adfunc)


def delrule(func):
    id = 'D_' + rid(func)
    res = getattr(forwardad, id)
    delattr(forwardad, id)
    return res


def restorerule(func):
    pass


def getrules():
    return forwardad.dict
