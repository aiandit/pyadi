from astunparse.astnode import ASTNode, Constant, Name
import random

class Assign(ASTNode):
    def __init__(self, l, r):
        self._class = 'Assign'
        if not isinstance(l, list):
            l = [l]
        self.targets = l
        self.value = r

class List(ASTNode):
    def __init__(self, elts):
        self._class = 'List'
        self.elts = elts

class Tuple(ASTNode):
    def __init__(self, elts):
        self._class = 'Tuple'
        self.elts = elts


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

class BinOp(ASTNode):
    def __init__(self, op, left=None, right=None):
        self._class = "BinOp"
        self.op = op
        self.left = left
        self.right = right

class AugAssign(ASTNode):
    def __init__(self, op, target=None, value=None):
        self._class = "AugAssign"
        self.op = op
        self.target = target
        self.value = value
