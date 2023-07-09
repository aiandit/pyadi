from itertools import chain

from .astvisitor import getmodule, rid
from . import forwardad


def getrule(func, adfunc, mode='D'):
    id = mode + '_' + rid(func)
    return getattr(forwardad, id)


def setrule(func, adfunc, mode='D'):
    id = mode + '_' + rid(func)
    setattr(forwardad, id, adfunc)


def delrule(func, mode='D'):
    id = mode + '_' + rid(func)
    res = getattr(forwardad, id)
    delattr(forwardad, id)
    return res


def getrules():
    return forwardad.__dict__
