from astunparse.astnode import ASTNode, BinOp, Constant, Name
import random

class Assign(ASTNode):
    def __init__(self):
        self._class = 'Assign'

class List(ASTNode):
    def __init__(self, elts):
        self._class = 'List'
        self.elts = elts

class Tuple(ASTNode):
    def __init__(self, elts):
        self._class = 'Tuple'
        self.elts = elts

class TmpVar(ASTNode):
    def __init__(self, kind='t'):
        self._class = 'TmpVar'
        self.id = random.random()
        self.kind = kind

class Module(ASTNode):
    def __init__(self, body):
        self._class = 'Module'
        self.body = body

class Subscript(ASTNode):
    def __init__(self, v, ind):
        self._class = "Subscript"
        self.value = v
        self.slice = Constant(ind)


class Call(ASTNode):
    def __init__(self, func, args=[], kw=[]):
        self._class = "Call"
        if isinstance(func, str):
            self.func = Name(func)
        else:
            self.func = func
        self.args = args
        self.keywords = kw


class Keyword(ASTNode):
    def __init__(self, arg, value):
        self._class = "Keyword"
        self.arg = arg
        self.value = value


class Starred(ASTNode):
    def __init__(self, value):
        self._class = "Starred"
        self.value = value


class UnaryOp(ASTNode):
    def __init__(self, op, value=None):
        self._class = "UnaryOp"
        self.op = op
        self.operand = value
