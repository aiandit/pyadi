from astunparse import Unparser
import sys, os, inspect, json
from io import StringIO

from astunparse import loadast, unparse2j, unparse
from astunparse.astnode import ASTNode, BinOp, Constant
from .astvisitor import ASTVisitorID

class ASTVisitorFMAD(ASTVisitorID):

    active_objects = ['self', 'dself', 'dt']
    active_fields = ['acc', 'vel', 'pos', 'axis']
    active_methods = ['equations']

    def ddispatch(self, tree):
        if isinstance(tree, list):
            return [self.ddispatch(t) for t in tree]
        if not isinstance(tree, ASTNode):
            return tree
        cname = tree._class
        meth = getattr(self, "_D"+cname, None)
        print(cname, vars(tree))
        tree = tree.clone()
        if meth:
            return meth(tree)
        else:
            for name in vars(tree):
                setattr(tree, name, self.ddispatch(getattr(tree, name)))
            return tree

    def _FunctionDef(self, t):
        if t.name in self.active_methods:
            t = t.clone()
            print(f'Catch Active FunctionDef {t.name} {vars(t)}')
            t.name = 'd_' + t.name
            t.args = self.ddispatch(t.args)
            nbody = []
            for item in t.body:
                if item._class == "Assign":
                    nbody += [ self.ddispatch(item.clone()) ]
                    nbody += [ self.dispatch(item) ]
                else:
                    nbody += [ self.dispatch(item) ]
            t.body = nbody
        else:
            t.args = self.dispatch(t.args)
            t.body = self.dispatch(t.body)
        return t

    def __Darguments(self, t):
        return t

    def _Darguments(self, t):
        assert(type(t.args) == type([]))
        dargs = []
        for t in t.args:
            print('   * args', t.arg, t.arg in self.active_objects)
            if t.arg in self.active_objects:
                tr1 = self.ddispatch(t.clone())
                tr2 = t.clone()
                dargs += [tr1, tr2]
            else:
                dargs += [t.clone()]
#        t.args = dargs
#        return t
        t.args = [d for d in dargs]
        return t

    def _Call(self, t):
        print(f'Catch Call {t.func} {vars(t)}')
        t.func = self.dispatch(t.func)
        t.args = self.dispatch(t.args)
        return t

    def _DCall(self, t):
        print(f'Diff Call {t.func} {vars(t)}')
        t = t.clone()
        t.func.attr = 'd_' + t.func.attr
        t.args = self.ddispatch(t.args)
        return t

    def _DName(self, t):
        print(f'Diff Name {t.id}')
        t = t.clone()
        t.id = 'd_' + t.id
        return t

    def _DAttribute(self, t):
        print(f'Diff Attribute {t.attr} of {vars(t.value)}')
        if not getattr(t.value, 'id', None):
            t.value = self.ddispatch(t.value.clone())
        else:
            if t.value.id in self.active_objects and t.attr in self.active_fields:
                t = t.clone()
                t.value.id = 'd_' + t.value.id
            else:
                t = Constant(0)
        return t

    def _DConstant(self, t):
        print(f'Diff Constant {t.value}')
        t.value = 0
        return t

    def _DBinOp(self, t):
        print(f'Diff BinOp {t.op} left {vars(t.left)}')
        if t.op == '*':
            left = BinOp('*')
            left.left = self.ddispatch(t.left.clone())
            left.right = self.dispatch(t.right.clone())
            right = BinOp('*')
            right.left = self.dispatch(t.left.clone())
            right.right = self.ddispatch(t.right.clone())
            t = BinOp('+')
            t.left = left
            t.right = right
        elif t.op == '+' or t.op == '-':
            t.left = self.ddispatch(t.left)
            t.right = self.ddispatch(t.right)
        else:
            t.left = self.dispatch(t.left)
            t.right = self.dispatch(t.right)
        return t

    def _BinOp(self, t):
        print(f'Catch BinOp {t.op}')
        t.left = self.dispatch(t.left)
        t.right = self.dispatch(t.right)
        return t


def diff2pys(intree, visitor):
    print('intree', unparse2j(intree), file=open('intree.json', 'w'))
    outtree = visitor(intree)
    print('outtree', unparse2j(outtree), file=open('outtree.json', 'w'))
    return unparse(outtree)

def diff2py(fname):
    with open(fname, "r") as pyfile:
        source = pyfile.read()
    return json2pys(source, fname)


def roundtrip2JIDs(source, fname):
    fmadtrans = ASTVisitorFMAD()
    fmadtrans.active_methods = ['equations']
    fmadtrans.active_fields = ['pos', 'vel', 'acc', 'axis']
    return diff2pys(loadast(source), fmadtrans)

def roundtrip2JID(fname):
    with open(fname, "r") as pyfile:
        source = pyfile.read()
        return roundtrip2JIDs(source, fname)

def diffmethod(obj, method, active=[]):
    meth = getattr(obj, method)
    csrc = inspect.getsource(meth).strip()
    fmadtrans = ASTVisitorFMAD()
    fmadtrans.active_methods = [method] + active
    fmadtrans.active_fields = ['position', 'speed', 'acceleration', 'axis'] + active
    fmadtrans.active_objects = ['self', 'dself', 'forces'] + active
    dsrc = diff2pys(loadast(csrc), fmadtrans)
    print(dsrc)
    gvars = {'data': {}}
    with open('diff.py', 'w') as f:
        f.write(dsrc)
    res = compile('import math\n' +
                  dsrc + '\ndata["d_equations"] = d_equations', 'diff.py', 'exec')
    exec(res, gvars)
    er = gvars['data']['d_equations']
    print('er', er)
    setattr(obj, 'd_' + method, er)
    return res

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

class Flywheel:
    def equations(self, dt):
        self.pos += self.vel * dt
        self.vel += self.acc * dt
                
def rundiffmethod():
    diffmethod(Flywheel, 'equations')
    fl = Flywheel()
    print(dir(fl))
    fl.d_equations()
    fl.equations()

if __name__ == "__main__":
    rundiffmethod()
    # testdir()
#run()
