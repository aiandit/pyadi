from astunparse.astnode import ASTNode, Constant, Name

class keyword(ASTNode):
    def __init__(self, arg, value):
        self._class = 'keyword'
        self.arg = arg
        self.value = value

class arguments(ASTNode):
    def __init__(self, args):
        self._class = 'arguments'
        self.args = args
        self.defaults = []
        self.vararg = None
        self.kwarg = None

class arg(ASTNode):
    def __init__(self, arg):
        self._class = 'arg'
        self.arg = arg
        self.annotation = None

class FunctionDef(ASTNode):
    def __init__(self, name, args, l):
        self._class = 'FunctionDef'
        self.name = name
        self.args = arguments([arg(n) for n in args])
        self.body = l
        self.decorator_list = []

class Lambda(ASTNode):
    def __init__(self, args, l):
        self._class = 'Lambda'
        self.args = arguments([arg(n) for n in args])
        self.body = l

class Return(ASTNode):
    def __init__(self, val):
        self._class = 'Return'
        self.value = val

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


class Slice(ASTNode):
    def __init__(self, l, u=None, s=None):
        self._class = 'Slice'
        self.lower = Constant(l) if isinstance(l, int) else l
        self.upper = Constant(u) if isinstance(u, int) else u
        self.step = Constant(s) if isinstance(s, int) else s


class Subscript(ASTNode):
    def __init__(self, v, ind):
        self._class = "Subscript"
        self.value = v
        if isinstance(ind, int):
            self.slice = Constant(ind)
        else:
            self.slice = ind


class Attribute(ASTNode):
    def __init__(self, v, attr):
        self._class = "Attribute"
        self.value = v
        self.attr = attr


class Call(ASTNode):
    def __init__(self, func, args=[], kw=[]):
        self._class = "Call"
        if isinstance(func, str):
            self.func = Name(func)
        else:
            self.func = func
        if not isinstance(args, list):
            args = [args]
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
