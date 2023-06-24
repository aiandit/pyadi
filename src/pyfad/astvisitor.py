from astunparse import Unparser
import sys, os, inspect, json
from io import StringIO
import time
import random

import astunparse
from astunparse import loadast, unparse2j
from astunparse.astnode import ASTNode, BinOp, Constant, Name, fields

def py(func):
    csrc = inspect.getsource(func).strip()
    return csrc

class ASTVisitor:

    def __init__(self):
        print('ASTVisitor()')

    def __call__(self, tree):
        print('ASTVisitor.call()')
        self.result = self.dispatch(tree)
        return self.result

    def dispatch(self, tree):
        print('ASTVisitor.dispatch()')
        if isinstance(tree, list):
            for t in tree:
                self.dispatch(t)
            return
        cname = tree._class
        meth = getattr(self, "_"+cname)
        meth(tree)

class ASTVisitorID(ASTVisitor):

    def dispatch(self, tree):
        if isinstance(tree, list):
            return [self.dispatch(t) for t in tree]
        if not isinstance(tree, ASTNode):
            return tree
        cname = tree._class
        meth = getattr(self, "_"+cname, None)
        print(cname, vars(tree))
        if meth:
            return meth(tree)
        else:
            for name in vars(tree):
                setattr(tree, name, self.dispatch(getattr(tree, name)))
            return tree

    def _BinOp(self, t):
        print(f'Catch BinOp {t.op}')
        t.left = self.dispatch(t.left)
        t.right = self.dispatch(t.right)
        return t


def isop(cn):
    return cn._class in ['BinOp', 'UnaryOp', 'BoolOp', 'AugAssign']

def iscall(cn):
    return cn._class in ['Call']

def iscanon(cn):
    return isop(cn) or iscall(cn)

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

tmpvars = {}
class TmpVar(ASTNode):
    def __init__(self, kind='t'):
        self._class = 'TmpVar'
        self.id = random.random()
#        print('****** TmpVar', self.id, self._class)
        self.kind = kind
        tmpvars[self.id] = self

class ASTCanonicalizer:
    def __init__(self):
        pass

    def __call__(self, tree):
        result = self.dispatch(tree)
        return result

    def edispatch(self, tree):
#        print('edisp', tree)
        if type(tree) == type([]):
            res = list(map(self.edispatch, tree))
        elif isinstance(tree, ASTNode):
            tmpas = Assign()
            tmpv = TmpVar()
            tmpas.targets = [tmpv]
            tmpas.value = self.dispatch(tree.clone())
            self._list.append(tmpas)
#            print('new tmp', repr(tmpas))
            res = tree
        else:
            res = tree
        return (res, tmpv)

    def dispatch(self, tree):
        if type(tree) == type([]):
            res = list(map(self.dispatch, tree))
        elif isinstance(tree, ASTNode):
#            print('visit', vars(tree))

            if tree._class == "FunctionDef":
#                print('canon', tree)
                nbody = []
                for stmt in tree.body:
                    self._list = []
                    pstmt = self.dispatch(stmt.clone())
                    nbody += self._list
                    nbody += [pstmt]
                tree.body = nbody
#                print('nbody:', astunparse.unparse(nbody))
                return tree

            for k in fields(tree):
                setattr(tree, k, self.dispatch(getattr(tree, k)))

            if tree._class == "BinOp":
                if iscanon(tree.left):
                    (tl, tmpvar) = self.edispatch(tree.left)
                    tree.left = tmpvar
                if iscanon(tree.right):
                    (tr, tmpvar) = self.edispatch(tree.right)
                    tree.right = tmpvar

            elif tree._class == "Call":
                nargs = []
                for arg in tree.args:
                    if iscanon(arg):
                        nargs += [self.edispatch(arg.clone())[1]]
                    else:
                        nargs += [arg]
                print('nargs', nargs)
                tree.args = nargs

            res = tree
        else:
            res = tree
        return res

def canonicalize(tree, **kw):
    an = ASTCanonicalizer()
    return an(tree)


class ASTReolvetmpvars:
    def __init__(self):
        pass

    def __call__(self, tree):
        result = self.dispatch(tree)
        return result

    def dispatch(self, tree):
        if type(tree) == type([]):
            res = list(map(self.dispatch, tree))
        elif isinstance(tree, ASTNode):

            res = tree
            for k in vars(tree).keys():
                setattr(res, k, self.dispatch(getattr(tree, k)))

            if tree._class == "TmpVar":
                res = Name(f'{tree.kind}_{int(tree.id * (1<<48)):d}')
        else:
            res = tree
        return res

def resolvetmpvars(tree, **kw):
    an = ASTReolvetmpvars()
    return an(tree)


def normalize(tree, **kw):
    tree = resolvetmpvars(tree)
    tree = astunparse.normalize(tree)
    return tree


class ASTFLocals:
    def __init__(self, tree):
        self.locals = []
        self.dispatch(tree)

    def dispatch(self, tree):
        if type(tree) == type([]):
            res = list(map(self.dispatch, tree))
        elif isinstance(tree, ASTNode):

            res = {k: self.dispatch(v) for k,v in fields(tree).items()}

            if tree._class == "Name":
                print('got Name', tree.id)
                self.locals.append(tree.id)
        else:
            res = tree
        return res

def locals(tree, **kw):
    if callable(tree):
        tree = loadast(py(tree))
    an = ASTFLocals(tree)
    return an.locals

def py2pys_check(jdict, visitor):
    if type(jdict) == type(''):
        jdict = json.loads(jdict)
    intree = JStructBuilder(jdict).result
    print(json.dumps(jdict, indent=1), file=open('in.json', 'w'))
    print(unparse2Jt(intree), file=open('out.json', 'w'))
    assert(json.loads(unparse2Jt(intree)) == jdict)
    outtree = visitor(intree)
    assert(json.loads(unparse2Jt(outtree)) == jdict)
    jbuf = StringIO()
    JStructUnparser(outtree, jbuf)
    return jbuf.getvalue()

def py2pys(jdict, visitor):
    if type(jdict) == type(''):
        jdict = json.loads(jdict)
    intree = JStructBuilder(jdict).result
    outtree = visitor(intree)
    jbuf = StringIO()
    JStructUnparser(outtree, jbuf)
    return jbuf.getvalue()

def py2py(fname):
    with open(fname, "r") as pyfile:
        source = pyfile.read()
    return py2pys(source, fname)


def roundtrip2JIDs(source, fname):
    return py2pys_check(py2jsons(source), ASTVisitorID())

def roundtrip2JID(fname):
    with open(fname, "r") as pyfile:
        source = pyfile.read()
        return roundtrip2JIDs(source, fname)

def run():
    fname = 'pyphy/parser.py'
    with open(fname) as f:
        code = f.read()
        output=sys.stdout
        tree = compile(code, fname, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
        c1 = ASTFragment(tree)
        Unparser(tree, output)

        src = inspect.getsource(Test.energy).strip()
        print(f'prop: "\n{src}"')
        roundtrip2Js(src, 'test_py')
#        tree = compile(src, 'test_py', "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
#        c2 = ASTFragment(tree)
#       UnparserJ(tree, output)

def testdir():
    base = 'examples'
    for name in sorted(os.listdir(base)):
        fname = os.path.join(base, name)
        if os.path.isfile(fname) and fname.endswith('.py'):
            with open(fname) as f:
                source = f.read()
                res = roundtrip2JIDs(source, fname)
                print(f"""File {fname}
Source:
{source}
Result:
{res}""")
                assert(py2jsons(source, 'a.py') == py2jsons(res, 'b.py'))

if __name__ == "__main__":
    testdir()
#run()
