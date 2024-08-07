import os
import inspect
import importlib
import warnings
import numpy as np

from itertools import chain

from astunparse import loadast, unparse2j, unparse2x, unparse
from astunparse.astnode import ASTNode, BinOp, Constant, Name, isgeneric, fields

from .astvisitor import canonicalize, resolvetmpvars, normalize, unnormalize, filterLastFunction
from .astvisitor import infoSignature, filterFunctions, py, getmodule, getast, fqname, fquname, fdname, fddname
from .astvisitor import ASTVisitorID, ASTVisitorImports, ASTVisitorLocals, mkTmp, isbuiltin
from .nodes import *
from .runtime import dzeros, unzd, joind, unjnd, DWith, lzip

from .runtime import binop_add, binop_sub, binop_mult, binop_c_mult, binop_d_mult, binop_matmult, binop_div, binop_floordiv, binop_mod, binop_pow
from .runtime import unaryop_uadd, unaryop_usub
from .runtime import augassign_add, augassign_sub, augassign_mult, augassign_div, augassign_truediv, augassign_mod
from .dtargets import mkActArgFunction, mkKwFunction

from .timer import Timer

from . import astvisitor


Debug = False

dpref_ = 'd_'

def setprefix(diff, tmp, common=''):
    global dpref_
    dpref_ = common + diff
    astvisitor.setprefix(common + tmp)


dumpDir = '.'
def dumpFile(fname): return os.path.join(dumpDir, fname)


def nodiff(tree):
    return tree._class == "Constant"


def isdiff(tree):
    return not nodiff(tree)


class ASTVisitorFMAD(ASTVisitorID):

    active_fields = []
    active_methods = []
    localvars = []
    verbose = 0

    def __call__(self, tree):
        """Process the tree. Calls dispatch, which will catch only
FunctionDefs and enter ddispatch traversal when the function is
designated as active.  Calls methods self._XYZ for individual node XYZ
handling, there is only _FunctionDef."""

        self.localvars, self.localfuncs = ASTVisitorLocals()(tree)
        self.active_methods += self.localfuncs
        if self.verbose > 2:
            print(f'Locals of {tree.name}', self.localvars)

        self.result = self.dispatch(tree)
        return self.result


    def dadispatch(self, tree):
        """traversal only for the LHS of assignments"""

        if isinstance(tree, list):
            return [self.dadispatch(t) for t in tree]
        elif isgeneric(tree):
            return tree
        cname = tree._class
        meth = getattr(self, "_Da_"+cname, None)
        # all nodes must be handled by a method
        assert meth, f'self.{"_Da_"+cname} not found'
        if meth:
            return meth(tree)
        else:
            res = ASTNode()
            for name in vars(tree).keys():
                delem = self.dadispatch(getattr(tree, name))
                setattr(res, name, delem)
            return res

    def _Da_Name(self, t):
        if not self.isLocal(t):
            return Name(dpref_ + '_')
        return self.ddispatch(t)

    def _Da_Attribute(self, t):
        if not self.isLocal(t):
            return Name(dpref_ + '_')
        return self.ddispatch(t)

    def _Da_Subscript(self, t):
        if not self.isLocal(t):
            return Name(dpref_ + '_')
        return self.ddispatch(t)

    def _Da_Tuple(self, t):
        t.elts = self.dadispatch(t.elts)
        return t


    def ddispatch(self, tree):
        """The main workhorse, the differentiation traversal
Calls methods self._DXYZ for individual node XYZ handling
"""
        if isinstance(tree, list):
            return [self.ddispatch(t) for t in tree]
        elif isgeneric(tree):
            return tree
        cname = tree._class
        meth = getattr(self, "_D"+cname, None)
#        print('ddispatch?', cname, vars(tree))
        if meth:
#            print('Found method', cname)
            return meth(tree)
        else:
#            print('start dispatch', vars(tree).keys())
#            print('start dispatch', dir(tree))
            res = ASTNode()
            for name in vars(tree).keys():
                delem = self.ddispatch(getattr(tree, name))
#                print(f'DDispatch {name} => {repr(delem)}')
                setattr(res, name, delem)
            return res


    nodiffFunctions = []
    nodiffExpr = ["Raise", "Assert"]
    def isnodiffExpr(self, item):
        res = False
        if item._class == "Expr":
            v = item.value
            if v._class == "Call":
                if getattr(v.func, 'id', False):
                    res = v.func.id in self.nodiffFunctions
        elif item._class in self.nodiffExpr:
            res = True
        return res


    def diffStmtList(self, body):
        nbody = []
        self.tmpval = None
        for item in body:
            if item._class == "Assign":
                atargets = [ s for s in item.targets if self.isLocal(s) ]
                if len(atargets) < len(item.targets):
                    natargets = [ s for s in item.targets if not self.isLocal(s) ]
                    warnings.warn(f'Assignment to non-local locations {[str(v).strip() for v in natargets]} cannot be handled, the derivative may be wrong')
                if len(atargets) > 0:
                    if item.value._class == "BinOp" and item.value.op == "**" and isdiff(item.value.right):
                        self.tmpval = mkTmp('s')
                        nbody += [Assign(self.tmpval, self.mkOpPartialC("**", None, None, item.value.left, item.value.right))]
                    nbody += [self.ddispatch(item.clone())]
                    if item.value._class == "BinOp" and item.value.op == "**" and isdiff(item.value.right):
                        nbody += [Assign(item.targets, self.tmpval)]
                        self.tmpval = None
                        continue
                    if item.value._class not in self.tupleDiff:
                        nbody += [self.dispatch(item)]
                else:
                    nbody += [self.dispatch(item)]
            elif item._class == "FunctionDef":
                nbody += [self._DFunctionDef(item.clone())]
                nbody += [item]
            elif item._class == "AugAssign":
                if item.op == "**" and isdiff(item.value):
                    self.tmpval = mkTmp('s')
                    nbody += [Assign(self.tmpval, self.mkOpPartialC("**", None, None, item.target, item.value))]

                if item.op in ['+', '-']:
                    if isdiff(item.value):
                        nbody += [self.ddispatch(item.clone())]
                elif item.op == '|':
                    nbody += [self.ddispatch(item.clone())]
                elif item.op == '//':
                    nbody += [Assign(self.ddispatch(item.target.clone()), Constant(0))]
                else:
                    if item.op == '*' or item.op == '@' or item.op == '/':
                        nbody += [AugAssign(item.op, self.ddispatch(item.target.clone()), item.value)]
                    elif item.op == '**':
                        nbody += [AugAssign('*', self.ddispatch(item.target.clone()), self.mkOpPartialL1(item.op, None, item.target, item.value))]
                    if isdiff(item.value):
                        op = '+'
                        rhspartial = self.mkOpPartialR(item.op, None, self.ddispatch(item.value.clone()), item.target, item.value)
                        if rhspartial._class == "UnaryOp":
                            op = rhspartial.op
                            rhspartial = rhspartial.operand
                        nbody += [AugAssign(op, self.ddispatch(item.target.clone()), rhspartial)]
                if item.op == "**" and self.tmpval is not None:
                    nbody += [Assign(item.target, self.tmpval)]
                else:
                    nbody += [item]
            elif self.isnodiffExpr(item):
                nbody += [self.dispatch(item)]
            else:
                nbody += [self.ddispatch(item)]
        return nbody

    def _FunctionDef(self, t):
        return self.ddispatch(t)

    def _DFunctionDef(self, t):
        if t.name in self.active_methods:
#            print(f'Catch Active FunctionDef {t.name} {vars(t)}')
            t.args = self.ddispatch(t.args)
            t.body = self.diffStmtList(t.body)
            prestmts = []
            if t.args.kwarg:
                prestmts += [Assign(Tuple([Name('d_' + t.args.kwarg.arg), Name(t.args.kwarg.arg)]), Call('unjnd', Name(t.args.kwarg.arg)))]
            if t.args.vararg:
                prestmts += [Assign(Tuple([Name('d_' + t.args.vararg.arg), Name(t.args.vararg.arg)]), Tuple([Subscript(Name(t.args.vararg.arg), Slice(0, s=2)), Subscript(Name(t.args.vararg.arg), Slice(1, s=2))]))]
            t.body = prestmts + t.body
            decos = []
            for d in t.decorator_list:
                if d._class == "Name":
                    decos += [
                        Lambda('f', Subscript(Call(Call('D', d), Tuple([Name('f'), Name(t.name)])), 0))
                    ]
                else:
                    decos += [
                        Lambda('f', Subscript(Call(Subscript(self.diffUnlessIsTupleDiff(d), 0), [Name('f'), Name(t.name)]), 0))
                    ]
            t.decorator_list = decos
            t.name = dpref_ + t.name
        else:
            t.args = self.dispatch(t.args)
            t.body = self.dispatch(t.body)
        return t

    def _DJoinedStr(self, t):
        t.values = [self.ddispatch(s) if s._class != "Constant" else s for s in t.values]
        return t

    def _DFormattedValue(self, t):
        t.value = self.diffUnlessIsTupleDiff(t.value)
        return t

    def _DIfExp(self, node):
        node.body = self.diffUnlessIsTupleDiff(node.body)
        node.orelse = self.diffUnlessIsTupleDiff(node.orelse)
        return node

    def _DGeneratorExp(self, t):
        t.elt = self.diffUnlessIsTupleDiff(t.elt)
        t.generators = self.ddispatch(t.generators)
        return t

    def _DLambda(self, node):
        bck = self.localvars
        self.localvars += ASTVisitorLocals.getVars(node.args.args)
        node.args = self.ddispatch(node.args)
        self.localvars = bck
        node.body = self.diffUnlessIsTupleDiff(node.body)
        return node


    tupleDiff = ["Call", "List", "ListComp", "Dict", "DictComp", "DictComp", "IfExp", "GeneratorExp", 'Starred']
    def diffUnlessIsTupleDiff(self, t, src=None):
        if t._class in self.tupleDiff:
            res = self.ddispatch(t.clone())
            if src and src._class == 'Call':
                if t._class == "List":
                    res = res.elts[0].value
            return res
        elif t._class == "Tuple":
            dargs = [self.diffUnlessIsTupleDiff(t) for t in t.elts]
            return Tuple([Starred(Call('zip', dargs))])
        else:
            return Tuple([self.ddispatch(t.clone()),t])

    def _DList(self, node):
        if len(node.elts):
            dargs = [self.diffUnlessIsTupleDiff(t) for t in node.elts]
            return List([Starred(Call('lzip', dargs))])
        return Tuple([List([]), List([])])

    def __DTuple(self, node):
        if len(node.elts):
            dargs = [self.diffUnlessIsTupleDiff(t) for t in node.elts]
            return Tuple([Starred(Call('zip', dargs))])
        return Tuple([List([]), List([])])

    def _DDict(self, node):
        node.values = [self.diffUnlessIsTupleDiff(t) for t in node.values]
        return Tuple([Starred(Call('unzd', node))])

    def _DDictComp(self, node):
        node.value = self.diffUnlessIsTupleDiff(node.value)
        node.generators = self.ddispatch(node.generators)
        return Call('unzd', [node])

    def _DListComp(self, node):
        node.elt = self.diffUnlessIsTupleDiff(node.elt)
        node.generators = self.ddispatch(node.generators)
        return Call('lzip', [Starred(node)])

    def _Dcomprehension(self, node):
        self._DForCommon(node)
        return node

    def _DIf(self, node):
        node.body = self.diffStmtList(node.body)
        node.orelse = self.diffStmtList(node.orelse)
        return node

    def _DWhile(self, node):
        node.body = self.diffStmtList(node.body)
        return node

    def _DForCommon(self, node):
        tnode = Tuple([self.ddispatch(node.target.clone()), node.target])
        if node.iter._class == "Call":
            itnode = Call('zip', [Starred(self.ddispatch(node.iter))])
        else:
            itnode = Call('zip', [self.ddispatch(node.iter.clone()), node.iter])
        node.target = tnode
        node.iter = itnode

    def _DFor(self, node):
        node.body = self.diffStmtList(node.body)
        self._DForCommon(node)
        return node

    def _Darguments(self, node):
        assert isinstance(node.args, list)
        dargs = []
        curargs = node.args
        for t in curargs:
            if t.arg in self.localvars:
                tr1 = self.ddispatch(t.clone())
                dargs += [tr1]
        node.args = list(chain(*zip(dargs, curargs)))
        ddefs = self.ddispatch(node.defaults)
        node.defaults = list(chain(*zip(ddefs, node.defaults)))
#        node.args = dargs + curargs
        return node

    nonder_builtins = ['next']

    nondercall_builtins = ['next']

    def _DCall(self, t):

        #print(f'Diff Call {t.func} {vars(t)}')
        dcallName = 'D'
        curargs = t.args
        if t.func._class == "Call" or self.isLocal(t.func):
            dcall = Call(Name('Dc'))
            dcall.args = [ self.diffUnlessIsTupleDiff(t.func) ]
        else:
            dcall = Call(Name('D'))
            dcall.args = [t.func]

        fcallname = getattr(t.func, 'id', '')
        if fcallname in self.nondercall_builtins:
            res = Call(fcallname)
        else:
            res = Call(dcall)

        dargs = [self.diffUnlessIsTupleDiff(a, t) for a in curargs]
        res.args = dargs
        res.keywords = self.diffKeywords(t.keywords)
        return res

    def diffKeywords(self, keywords):
        res = []
        if len(keywords) == 0:
            return res
        for key in keywords:
            if key.arg is None:
                res += [ self.diffUnlessIsTupleDiff(key.value) ]
            else:
                res += [ Call('unzd', Call('dict', keyword(key.arg, self.diffUnlessIsTupleDiff(key.value)))) ]
        # print('keywords::', res)
        zcall = Call('joind', Starred(Call('zip', res)))
        return [keyword(None, zcall)]

    def _Darg(self, t):
        if t.arg in self.localvars:
            t.arg = dpref_ + t.arg
            #print('   * active arg', t.arg)
        return t

    def _Dkeyword(self, t):
        t.arg = dpref_ + t.arg
        t.value = self.ddispatch(t.value)
        return t

    def _DAssign(self, t):
        atargets = [ s for s in t.targets if self.isLocal(s) ]
        if t.value._class in self.tupleDiff:
            t.targets = [Tuple(self.dadispatch([t.clone() for t in atargets]) + self.dispatch(t.targets))]
        else:
            #t.targets = [self.ddispatch(s.clone()) if self.isLocal(s) else Name('_') for s in t.targets ]
            t.targets = [self.dadispatch(s.clone()) for s in atargets ]
        isList = t.value._class == "List" or t.value._class == "Dict"
        t.value = self.ddispatch(t.value)
        if isList:
            if hasattr(t.value, 'elts') and t.value.elts[0]._class == "Starred":
                t.value = t.value.elts[0].value
        return t

    def _DName(self, t):
        #print(f'Diff Name {t.id}')
        if t.id in self.localvars:
            t.id = dpref_ + t.id
            return t
        return Call('dzeros', t)

    def getRoot(self, t):
        if t._class == "Attribute" or t._class == "Subscript":
            return self.getRoot(t.value)
        return t

    def isLocal(self, t):
        if t._class == "Tuple":
            return any([self.isLocal(s) for s in t.elts])
        return getattr(self.getRoot(t), 'id', '') in self.localvars

    def _DStarred(self, node):
        node.value = Call('zip', [Starred(self.diffUnlessIsTupleDiff(node.value))])
        return node

    def _DSubscript(self, node):
        if not self.isLocal(node.value):
            return Call('dzeros', node)
        node.value = self.ddispatch(node.value)
        return node

    def _DAttribute(self, t):
        #print(f'Diff Attribute {t.attr} of {vars(t.value)} {self.imports}')
        if not t.value._class == "Call" and not self.isLocal(t.value):
            return Call('dzeros', t)
        t.value = self.ddispatch(t.value)
        return t

    def _DConstant(self, t):
        if isinstance(t.value, float) or isinstance(t.value, int):
            t = t.clone()
            t.value = 0
        elif isinstance(t.value, complex):
            t = t.clone()
            t.value = 0j
        return t

    def mkOpPartialC(self, op, r, dx, x, y):
        if op == '**':
            if r is None:
                t = BinOp(op, x, y)
            else:
                t = r
        return t

    def mkOpPartialL1(self, op, r, x, y):
        if op == '**':
            if y._class == "Constant":
                if y.value == 2:
                    t = BinOp('*', y, x)
                else:
                    t = BinOp('*', y, BinOp('**', x, Constant(y.value -1)))
            else:
                t = BinOp('*', y, BinOp('**', x, BinOp('-', y,  Constant(1))))
        else:
            raise ValueError()
        return t

    def mkOpPartialL(self, op, r, dx, x, y):
        if op == '/':
            t = BinOp('/', dx, y)
        elif op == '%':
            t = dx
        elif op == '**':
            p1 = self.mkOpPartialL1(op, r, x, y)
            t = BinOp('*', p1, dx)
        return t

    def mkOpPartialR(self, op, r, dy, x, y):
        if op == '*' or op == '@':
            t = BinOp(op, x, dy)
        elif op == '/':
            sq = BinOp('**', y, Constant(2))
            right_ = BinOp('*', x, dy)
            t = UnaryOp('-', BinOp('/', right_, sq))
        elif op == '%':
            quot = BinOp('/', x, y)
            t = UnaryOp('-', BinOp('*', Call('math.floor', [quot]), dy))
        elif op == '**':
            t = BinOp('*', Call('log', [x]), dy)
            t = BinOp('*', self.tmpval if self.tmpval is not None else BinOp('**', x, y), t)
        return t

    def _DBinOp(self, t):
        #print(f'Diff BinOp {t} left {vars(t.left)}')
        if nodiff(t.left) and nodiff(t.right):
            return Constant(0)

        if nodiff(t.left):
            left = t.left
        else:
            left = self.ddispatch(t.left.clone())
        if nodiff(t.right):
            right = t.right
        else:
            right = self.ddispatch(t.right.clone())

        if t.op == '*' or t.op == '@':

            if isdiff(t.left) and isdiff(t.right):
                left_ = BinOp(t.op, left, t.right)
                right_ = BinOp(t.op, t.left, right)
                t = BinOp('+', left_, right_)
            else:
                t.left = left
                t.right = right

        elif t.op == '/':
            if isdiff(t.right):
                right_ = self.mkOpPartialR('/', None, right, t.left, t.right)
                if isdiff(t.left):
                    left_ = self.mkOpPartialL('/', None, left, t.left, t.right)
                    t = BinOp('-')
                    t.left = left_
                    t.right = right_.operand
                else:
                    t = right_
            elif isdiff(t.left):
                t.left = left

        elif t.op == '//':
            t = Constant(0)

        elif t.op == '%':
            if t.right._class == "Tuple" or (t.left._class == "Constant" and isinstance(t.left.value, str)):
                return t
            if isdiff(t.left):
                lfact_ = self.mkOpPartialL('%', None, left, t.left, t.right)
            if isdiff(t.right):
                rfact_ = self.mkOpPartialR('%', None, right, t.left, t.right)
                if isdiff(t.left):
                    t = BinOp('+')
                    t.left = lfact_
                    t.right = rfact_
                else:
                    t = rfact_
            elif isdiff(t.left):
                t = lfact_

        elif t.op == '**':
            if isdiff(t.left):
                lder = self.mkOpPartialL('**', None, left, t.left, t.right)

            if isdiff(t.right):
                rder = self.mkOpPartialR('**', None, right, t.left, t.right)

                if isdiff(t.left):
                    term = BinOp('+', lder, rder)
                else:
                    term = rder

            elif isdiff(t.left):
                term = lder

            t = term

        elif t.op == '+' or t.op == '-':
            if nodiff(t.left):
                left = self.ddispatch(t.left.clone())
            if nodiff(t.right):
                right = self.ddispatch(t.right.clone())

            t.left = left
            t.right = right

        else:
            t.left = self.dispatch(t.left)
            t.right = self.dispatch(t.right)
        return t

    def _DReturn(self, t):
        t.value = self.diffUnlessIsTupleDiff(t.value)
        return t

    def _DYield(self, t):
        t.value = self.diffUnlessIsTupleDiff(t.value)
        return t

    def _DTry(self, t):
        t.body = self.diffStmtList(t.body)
        t.handlers = self.ddispatch(t.handlers)
        return t

    def _DExceptHandler(self, t):
        t.body = self.diffStmtList(t.body)
        return t

    def _DWith(self, t):
        t.items = self.ddispatch(t.items)
        t.body = self.diffStmtList(t.body)
        return t

    def _Dwithitem(self, t):
        t.context_expr = Call('DWith', self.diffUnlessIsTupleDiff(t.context_expr))
        if t.optional_vars:
            t.optional_vars = self.diffUnlessIsTupleDiff(t.optional_vars)
        return t

    def _DDelete(self, t):
        dtargets = [ self.ddispatch(s.clone()) for s in t.targets if self.isLocal(s) ]
        t.targets = dtargets + t.targets
        return t

    def _DNonlocal(self, t):
        dtargets = [ 'd_' + s for s in t.names ]
        t.names = dtargets + t.names
        return t


def diff2pys(intree, visitor, **kw):
    # print('intree', unparse2j(intree, indent=1), file=open('intree.json', 'w'))

    if kw.get('verbose', 0) > 2:
        print(f'Input code for {getattr(intree, "name", "")}:', unparse(intree))

    intree = normalize(intree.clone(), **kw)

    intree = canonicalize(intree)

    if kw.get('verbose', 0) > 1:
        print(f'Preprocessed code for {getattr(intree, "name", "")}:', unparse(intree))

    outtree = visitor(intree)

    outtree = unnormalize(outtree.clone(), **kw)

    return outtree


def differentiate(intree, activef=None, active=None, modules=None, filter=False, prefix=None, **kw):
    fmadtrans = ASTVisitorFMAD()

    fmadtrans.verbose = kw.get('verbose', 0)

    if prefix:
        while len(prefix) < 3:
            prefix.append('')
        setprefix(*prefix)

    if modules is None:
        _, modules = ASTVisitorImports()(intree)

    fmadtrans.imports = modules
    # print('imports', fmadtrans.imports)
    # print('source', unparse(intree))

    if activef is None:
        intree, fname = filterLastFunction(intree)
        fmadtrans.active_methods = [fname]
    else:
        fmadtrans.active_methods = varspec(activef)
        if filter:
            intree = filterFunctions(intree, activef)

    if active is None or len(active) == 0:
        fname, sig = infoSignature(intree)
        # fmadtrans.active_fields = [sig[0]]
        fmadtrans.active_fields = sig
    else:
        fmadtrans.active_fields = varspec(active)

    dtree = diff2pys(intree, fmadtrans, **kw)
    return dtree


def diff2py(fname):
    with open(fname, "r") as pyfile:
        source = pyfile.read()
    return unparse(diff2pys(source, fname))


def diff2pys2s(source, fname):
    return unparse(differentiate(loadast(source)))


def execompile(source, fglobals={}, flocals={}, imports=['math', 'sys', 'os', {'pyadi': 'D'}], vars=['x'], fname='', **kw):

    # importstr = '\n'.join([f'import {name}' if isinstance(name, str)
    #                        else ('\n'.join([f'from {k} import {v}' for k, v in name.items()])) for name in imports])
    collectstr = '\n'.join([f'_pyadi_data["{name}"] = {name}' for name in vars])

    dsrc = f"{source}\n{collectstr}"

    try:
        res = compile(dsrc, fname, "exec")
    except SyntaxError as ex:
        # print(f'Compilation error in diff source:\n{ex}')
        raise ex

    gvars = globals() | fglobals | {'_pyadi_data': {}}
    #print(f'exec compiled diff function code in file {sfname} with globals={(gvars).keys()} locals={flocals}')
    exec(res, gvars, flocals)

    result = {name: gvars["_pyadi_data"][name] for name in vars}
    gvars |= result
    return result


def Dpy(func, active=[], **kw):
    csrc, imports, modules = getast(func, **kw)
    dtree = differentiate(csrc, activef=func.__name__, active=active, modules=modules, **kw)
    return dtree


def mkClosDict(function):
    """Return dictionary of closure variables of ``function``."""
    clos = getattr(function, '__closure__', None)
    cl_data = {}
    if clos is not None:
        code = function.__code__
        cl_data = { code.co_freevars[i]: clos[i].cell_contents for i in range(len(clos)) }
    return cl_data


def difffunction(func, active=[], **kw):
    dtree = Dpy(func, active, **kw)

    try:
        dsrc = unparse(dtree)
    except BaseException as ex:
        print(unparse2j(dtree, indent=1), file=open('d_failed.json', 'w'))
        print(unparse2x(dtree, indent=1), file=open('d_failed.xml', 'w'))
        print(f"""Failed to unparse diff code, exception:
{ex}
Source:
{py(func)}
""")
        raise ex

    if kw.get('verbose', 0) > 1:
        print(f'Diff code {fdname(func)}:{dsrc}')

    sfname = ''
    if kw.get('dump', 0) > 0:
        sfname = dumpFile(fddname(func) + '.py')
        print(dsrc, file=open(sfname, 'w'))

    cl_data = mkClosDict(func)
    if cl_data and kw.get('verbose', 0) > 1:
        print(f'difffunction: Function {func} has a closure: {cl_data.keys()}')

    fkey = dpref_ + func.__name__
    # globals = func.__globals__ if not isinstance(func, type) else func.__init__.__globals__
    gvars = func.__globals__ | kw.get('globals', {}) | cl_data

    try:
        dfunc = execompile(dsrc, vars=[fkey], fglobals=gvars, fname=sfname, **kw)
    except BaseException as ex:
        print(unparse2j(dtree, indent=1), file=open('d_failed.json', 'w'))
        print(unparse2x(dtree, indent=1), file=open('d_failed.xml', 'w'))
        print(dsrc, file=open('d_failed.py', 'w'))
        print(f"""Failed to compile diff code, exception:
{ex}
Diff code:
{dsrc}
Source:
{py(func)}
""")
        raise ex

    dfunc = dfunc[fkey]
    global adglobalsc
    adglobalsc[fqname(func)] = dfunc.__globals__
    return (dfunc, active)

adglobalsc = {}

def fid(func, active):
    mod, modfile = getmodule(func)
    if modfile is None:
        modfile = mod
    fid = f'{func.__qualname__}:{modfile}:{repr(active)}'
#    print('FID', func, fid)
    return fid


def getsig(f):
    x = inspect.signature(f)
    x = [f for f in x.parameters]
    return x


def varspec(x):
    if isinstance(x, str):
        x = x.split(',')
    assert isinstance(x, list)
    return x


def parinds(f, x):
    x = varspec(x)
    if len(x) > 0 and isinstance(x[0], str):
        sig = getsig(f)
        inds = [sig.index(a) for a in sig if a in x]
    else:
        inds = x
    return inds


adc = {}


def clear(search=None):
    global adc
    if search is None:
        adc = {}
        astvisitor.modastcache = {}
        astvisitor.getast(mkActArgFunction)
    else:
        if search in adc:
            del adc[fqname(search)]


def doSourceDiff(function, opts):

    # Try source diff
    adfun = None
    _class = None

    # print(f'SD: {function.__name__}')

    if isbuiltin(function):
        fname = fqname(function)
        id = astvisitor.rid(function)
        msg = f'No rule for buitin {fname}, function {id} not found'
        raise (NoRule(msg))

    elif isinstance(function, type):
        # print(f'Cannot diff. a type! {function.__name__}')
        return mkConstr(function)

    (adfun, actind) = difffunction(function, **opts)

    return adfun

rulemodules = {}
def clearrulemodules(name=None):
    global rulemodules
    rulemodules = {}
def addrulemodule(module, **kw):
    deco = module.decorator(**kw)
    handle = None
    if isinstance(deco, tuple):
        deco, handle = deco
    alias = kw.get('alias', module.__name__)
    ind = 0
    while f'{alias}{ind}' in rulemodules:
        ind += 1
    if ind == 0: ind = ''
    rulemodules[f'{alias}{ind}'] = module, deco, handle

def initRules(rules='ad=pyadi.forwardad', **opts):
    """Initialize the rule processing mechanism for
    :py:func:`processRules` that performs the mapping of functions to
    differentiated functions.

    The :py:func:`.decorator` of a rule module may return two function
    handles instead of one. In this case the second one can be
    retrieved using :py:func:`.getHandle`, possibly to manipulate the
    scope of the returned differentiated functions at runtime, as
    demonstrated by the :py:func:`~.trace.decorator` of the rule
    module :py:mod:`.trace`.

    When the same module shall be used several times in the chain, an
    alias can be defined, for example::

      pyadi.initRules(
        rules='pyadi.trace,pyadi.forwardad,tr2=pyadi.trace',
        tracecalls=True, verbose=True, verboseargs=True)

    Then, ``getHandle('tr2')`` retrieves the handle to the second
    instance of the decorator that the trace module installed, and
    ``getHandle('pyadi.trace')`` gets that of the first, while
    ``getHandle('pyadi.forwardad')`` is None because it does not
    provide a handler function.

    Parameters
    ----------
    rules : str
        Comma-separated list of python modules to use as rule
        modules. Entries can use 'alias=module' to define a name
        alias.

    opts : dict
        Passed to :py:func:`decorator` of all the rule modules upon
        initialization.

    """
    clearrulemodules()
    rules = rules.split(',')
    for rule in rules:
        add = {}
        if '=' in rule:
            (alias, rule) = rule.split('=')
            add['alias'] = alias
        rmod = importlib.import_module(rule)
        addrulemodule(rmod, **add, **opts)

initRules()


def getHandle(alias):
    """Return handle to a rule module.

    Return the second item of the result of a rule module's decorator
    function, or None. This second item is meant to be a function that
    can manipulate or read the local scope of the decorator, for an
    example see the tracing rule module :py:mod:`.trace`.

    The result returned becomes invalid whenever :py:func:`initRules`
    is called again.

    Parameters
    ----------
    alias : str
       The module name or alias used with :py:func:`initRules`.


    Returns
    -------
    object, usually function or None
      The second item that the rule module's decorator returned,
      usally a second inner function.

    """
    return rulemodules[alias][2]


def callHandle(name, *args, **kw):
    results = [mod[2](*args, **kw) for alias, mod in rulemodules.items()
               if mod[0].__name__ == name]
    return results


def getRuleModules(index):
    return rulemodules


class NoRule(BaseException):
    pass

def processRules(function, opts, *args, **kw):
    mkeys = list(rulemodules.keys())

    def nextStep(ind=0):
        if ind >= len(mkeys):
            dres = doSourceDiff(function, opts, *args, **kw)
        else:
            deco = rulemodules[mkeys[ind]][1]
            dres = deco(nextStep, ind+1, function, *args, **kw)
        #print(f'nextStep({ind}): {function.__name__} = {dres}')
        return dres
    return nextStep()


def initType(function, *args, **kw):
    """Create pair of objects d_o, and o by calling the constructor
    ``function`` twice, and then zero all floats in d_o with
    :py:func:`dzeros`.

    """
    do, o = function(*args[1::2], **kw), function(*args[1::2], **kw)
    do = dzeros(do)
    return do, o

def mkConstr(function):
    def constr(*args, **kw):
        args = list(chain(*args))
        return initType(function, *args, **kw)
    return constr

class GenIter:
    def __init__(self, genobj):
        self.genobj = genobj

    def __iter__(self):
        self.index = 0
        self.findex = 0
        return self

    def next(self):
        self.nitem = next(self.genobj)
        self.index += 1

    def __next__(self):
        if self.index == self.findex:
            # print('l')
            self.next()
        # else: assert self.findex +1 == self.index
        self.findex += 1
        # assert self.findex == self.index
        return self.nitem[0]

class GenIter2:
    def __init__(self, genobj):
        self.genobj = genobj

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.genobj.index == self.index:
            print('r')
            assert False
            # is it true this never happens?
            next(self.genobj)
        # else: assert self.index +1 == self.genobj.index
        self.index += 1
        # assert self.index == self.genobj.index
        return self.genobj.nitem[1]


def doDiffFunction(function, **opts):
    """Produce differentiated functions.

    This function is called to produce a derivative function for
    ``function``, that is, a function that is called with tuples (dx, x)
    for each original argument x in the original code, and that
    returns a tuple (dr, r) where r is the function result and dr is
    the derivative.

    This function will call :py:func:`processRules`, which calls the
    installed rule modules.

    The default rule module :py:mod:`.forwardad` will for example
    catch calls to :py:func:`print`, which is a builtin function, and
    call :py:func:`.mkRule` with :py:func:`.D_builtins_print` to
    produce a suitable result, namely, it will print the
    differentiated arguments in an additional line to the original
    print, which in the case of a formatted :py:doc:`f-string` would be
    the same f-string, but with differentiated expressions..

    When no rule module catchs the call and returns a suitable
    function, finally the source differentiation
    :py:func:`doSourceDiff` is invoked. It will produce a suitable
    result by retrieving the AST of function using :py:func:`.getast`
    and differentiating that.

    A few special cases need to by handled as follows:

       - When function is a type, then a constructor is being called.

          - set _C = function

          - When _C.__init__ is not builtin, set function =
            _C.__init__, that is the constructor class method.

       - When function has a closure and the last closure entry
         fdec is a function this might be a call to a decorated
         function, that is, the result of deco(fdec), which is a
         local function that captured fdec. So when fdec has a
         decorator list (getting the ast of fdec), this might be
         the right thing. TODO: We should check if this is really
         the right function. However, substitute function by
         fdec. This will thus in the following get the AST of fdec,
         which is something like::

             @mydeco2(1.23)
             def gdeco2(l):
                 return gl_sum(l)

         This gets differentiated to an expression decorating the
         regular D(fdec) with the differentiated decorator
         expression::

             @(lambda f: D(mydeco2)((0, 1.23))[0](f, gdeco2)[0])
             def d_gdeco2(d_l, l):
                 return D(gl_sum)((d_l, l))

         Which when loaded produces the differentiated inner function
         that the differentiated decorator, that D(mydeco2) returns,
         creates when called with d_gdeco2 alias f, and gdeco2. Thus,
         whatever happens in these decorators and the inner functions
         that they produce is getting differentiated regulary.

         The differentiated decorator expression is one of the few
         cases where we have to throw away the second part of the
         tuples that differentiated functions produce, because we only
         need the differentiated result, and even twice in this case.

    Then the differentiated function ``adfun`` is produced by calling
    :py:func:`.processRules`. This function returns however a local
    function ``def theADFun(*args, **kw):`` that does the following:

      - first flatten the argument list ``args`` of N tuples to a list
        of 2*N, alternating derivative and regular arguments. This is
        because the source differentiation differentiates ``def f(x,
        y):`` to ``def d_f(d_x, x, d_y, y):``. Hence, the builtin
        rules will also be called with the flattened list of
        arguments. This step also forces the evaluation of potentially
        lazy zip and other iterators that the arguments may be.

      - when ``_C`` is not None, a type, that is, a constructor has
        been called. Initialize two objects d_o and o with
        :py:func:`.initType`.  Prepend ``(d_o, o)`` to the list of
        arguments. This requires that _C.__init__ accepts being called
        with no arguments. Both objects will then be reinitialized
        again with the provided arguments to the constructor when the
        differentiated __init__ method ``adres`` is invoked in the
        next step. TODO: check if we can somehow produce unitialized
        objects, that is, do what Python does before it calls
        __init__?

      - Another corner case is when ``function`` is not a function but
        a bound method. This can only happen with global objects being
        called, like the method ``get`` of :py:obj:`os.environ`. Then
        the self pointer ``o`` is extracted and a copy ``d_o`` to be
        :py:func:`.dzeros`-ed must be created somehow on the
        fly. Prepend ``(d_o, o)`` to the list of arguments.

      - Finally call ``adres = adfun(*args, **kw)``

      - when ``adres`` is None, which often happens with methods, return
        ``(None, None)``, unless ``_C`` is not None, then a constructor has
        been called, return ``(d_o, o)``.

      - when ``adres`` is an object of the builtin type ``generator``,
        ``function`` was a generator function and ``adfun`` is too,
        with differentiated yield statements that produce
        tuples. Create two coupled iterators ``d_it =``
        :py:class:`GenIter` (``adres``) and ``it =``
        :py:class:`GenIter2` (``d_it``) that in tandem iterate
        ``adres``, one returning ``r[0]`` and the other ``r[1]`` of
        the tuples ``r`` produced, and return ``(d_it, it)``.

      - otherwise, return ``adres``, which is a tuple.

    """
    _class, constr, deco = None, None, None

    if isinstance(function, type):
        # print(f'SD: {function.__name__} is a type!')
        _class = function
        if not isbuiltin(function.__init__):
            constr = function = function.__init__
        else:
            #print(f'SD: type {function.__name__} has a builtin constructor !')
            pass
    else:
        clos = getattr(function, '__closure__', None)
        if clos is not None and len(clos) > 0:
            if isinstance(clos[-1].cell_contents, type):
                # print(f'Function {function} is a method, {function.__closure__[-1]}')
                pass
            elif callable(clos[-1].cell_contents) and len(getast(clos[-1].cell_contents)[0].decorator_list) > 0:
                # print(f'Function {function} has a closure and is decoratored, {function.__closure__[-1]}')
                deco = function
                function = clos[-1].cell_contents
            else:
                # handled only in case of source diff, later
                pass

    self = getattr(function, '__self__', None)
    if self is not None:
        selfClass = self.__class__
        if selfClass.__name__ == 'module':
            self = None


    adfun = processRules(function, opts)
    if opts.get('verbose', 0):
        print(f'AD function produced for {fqname(function)}: {adfun.__qualname__}')

    def theADFun(*ADargs, **kw):

        args = list(chain(*ADargs))
        #print(f'adfun called for {function.__qualname__}: {adfun.__qualname__}: {ADargs}, kw={kw}')

        if constr is not None:
            #print(f'adfun called for constr {function.__qualname__}: {adfun.__qualname__}, kw={kw}')
            d_kw, f_kw = unjnd(kw)
            do, o = initType(_class, *args, **f_kw)
            args = [do, o] + list(args)
        elif self is not None:
            try:
                dself = dzeros(self.__class__())
            except:
                dself = dzeros(self)
            args = [dself, self] + list(args)

        adres = adfun(*args, **kw)
        if adres is None:
            if _class:
                # was constructor
                adres = do, o
            else:
                adres = None, None
        elif adres.__class__.__name__ == 'generator':
            adres_d = GenIter(adres)
            adres_v = GenIter2(adres_d)
            adres = adres_d, adres_v

        return adres

    return theADFun


def DiffFunction(function, **opts):
    """Runtime decorator to handle function calls.

    This function merely caches the calls to
    :py:func:`doDiffFunction`, which does the actual work when no
    entry is found for function.

    Use :py:func:`clear` to clear this cache, which should be
    necessary only when the processing is redefined at runtime using
    :py:func:`initRules`.

    """
    ckey = fquname(function)
    centry = adc.get(ckey, None)
    if centry is None:
        # print(f'Diff function {fqname(function)}')
        adfun = doDiffFunction(function, **(transformOpts|opts))
        adc[ckey] = (adfun, function)
        # print(f'Diff function {function.__name__} cached => {adfun.__name__}')
    else:
        adfun = centry[0]
        if opts.get('verbose', 0) > 2:
            print(f'Found diff function {fqname(function)} in cache: {adfun.__name__}')

    cl_data = mkClosDict(function)
    if cl_data:
        if opts.get('verbose', 0) > 1:
            print(f'DiffFunction: Function {function} has a closure: {cl_data.keys()}')
        if fqname(function) in adglobalsc:
            adglobalsc[fqname(function)] |= cl_data

    return adfun


D = DiffFunction
"""An alias for :py:func:`.DiffFunction` so the generated code can be shorter."""


def DiffFunctionObj(tpl, **opts):
    """Runtime decorator to handle calls to local variables.

    Calls to local variables like ``obj.meth`` are differentiated to
    an expression invoking this function as ``Dc((d_obj.meth,
    obj,meth))``, that is, with a tuple of the "differentiated"
    function and the original function. Differentiated is in quotes
    because different cases can happen.

    Let the tuple tpl be expanded to dfunc, func.

    This function will usually call DiffFunction (aka. D) with func
    after handling the following cases:

      - When dfunc != func:

        1) A method is being called, dfunc and func are bound
           methods. Extract the two self pointers from both and
           substitute func with the actual class function T.meth,
           where T is the type. T is not necessarily the type of obj
           but may also be a parent class when an inherited method is
           being called using super().

           At runtime, inject the two self pointers to the front of
           the argument list.

        2) When func is not a function, then an object is being
           called, dfunc is the derivative object. Substitute func by
           T.__call__, where T is the type of obj.

           At runtime, inject the two self pointers (that is, dfunc
           and the original func) to the front of the argument list.

        3) Otherwise, a local function inner is being called, which
           will have been differentiated in source code already, the
           call to this decorator then being Dc(d_inner,
           inner). Hence, dfunc is already the differentiated function
           of func, it can be called directly. In this case D() is not
           called in the following.

      - When dfunc == func: A function alias has been called, that is,
        a global function was assigned to a local variable like myf =
        math.sin. The differentiated variable d_myf then has
        dzeros(math.sin), which is also math.sin. Do nothing.

      . Finally call DiffFunction(func) and return that result, unless
        the runtime arguments need patching, then return a local
        function doing that.

    """
    dfunc, function = tpl

    dself, self = None, None
    adfun = None

    # print(f'diff likely method {function}: {dfunc}: {opts}')
    if dfunc != function:
        self = getattr(function, '__self__', None)
        # print(f'different functions {function}: {dfunc}: self {self}')
        if self is not None:
            if self.__class__.__name__ != 'module':
                parts = function.__qualname__.split('.')
                cname = parts[-2]
                # pick the right class from object's MRO
                _class = [c for c in self.__class__.__mro__ if c.__name__ == cname][0]
                dself = dfunc.__self__
                function = getattr(_class, function.__name__)

        elif not hasattr(function, '__qualname__'): # not isinstance(function, Function):
            _class = function.__class__
            dself, self = dfunc, function
            function = function.__class__.__call__
        else:
            def inner(*args, **kw):
                # print(f'inner shortcut called: {dfunc.__qualname__} for {function.__qualname__}')
                args = list(chain(*args))
                return dfunc(*args, **kw)
            return inner

    if dself is not None:
        dfname = f'd_{function.__name__}'
        #adfun = getattr(_class, dfname, None)

    # print(f'DC: {dfunc} for {function} self={self}, dself={dself}')
    if adfun is None:
        adfun = DiffFunction(function, **opts)
        if dself is not None:
            try:
                setattr(_class, dfname, adfun)
                # print(f'Diff function {function.__name__} saved class type as {dfname} => {adfun.__name__}')
            except:
                pass
    else:
        # print(f'Diff function {function.__name__} in class type as {dfname} => {adfun.__name__}')
        pass

    def inner(*args, **kw):
        # Prepend dself and self to method call
        # print(f'method called: {adfun.__qualname__} for {function.__qualname__}, kw={kw}')
        return adfun((dself, self), *args, **kw)

    return inner if dself is not None else adfun

Dc = DiffFunctionObj
"""An alias for :py:func:`DiffFunctionObj` so the generated code can be shorter."""


def nvars(args):
    """Compute recursively the total number of values in the list args."""
    if isinstance(args, list) or isinstance(args, tuple):
        return sum([nvars(f) for f in args])
    elif isinstance(args, dict):
        return sum([nvars(v) for f, v in args.items()])
    elif isgeneric(args):
        return 1
    elif hasattr(args, 'flat'):
        return args.size
    else:
        return len(args)


def varv(args):
    if isinstance(args, list) or isinstance(args, tuple):
        return chain(*[varv(f) for f in args])
    elif isinstance(args, dict):
        return chain(*[varv(v) for f, v in args.items()])
    elif isgeneric(args):
        return [args]
    elif hasattr(args, 'flat'):
        return list(args.flat)



class FillHelper:
    """A simple iterator used to source floats one by one from either
    a list or a :py:mod:`numpy` array, used by :py:func:`.fill`.

    """
    def __init__(self, seed):
        self.seed = np.array(seed)
        self.len = nvars(seed)
        self.offs = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Return a single float from the data.
        """
        if self.offs < self.len:
            r = self.seed[self.offs]
            self.offs += 1
            return r
        else:
            raise StopIteration

    def __repr__(self):
        """Print the fill status like "FillHelper(i/n)".
        """
        return f'FillHelper({self.offs}/{len(self.seed)})'

    def get(self, N):
        """Batch-return N values to speed up filling arrays."""
        r = self.seed[self.offs:(self.offs+N)]
        self.offs += N
        return r


def fill(arg, seed):
    """Fill arg with values from seed.

    Fill arguments arg with values from seed. Lists, tuples, dicts and
    objects are deep-copied and each generic value, as per
    :py:func:`astunparse.astnode.isgeneric` is filled with one value
    from seed using :py:class:`.FillHelper`.  :py:mod:`numpy` arrays
    are batch-filled in-place using :py:meth:`.FillHelper.get`, so
    :py:func:`.dzeros` should be used before if it is desired that
    arrays are cloned and the original arrays not modified

    Parameters
    ----------
    arg : list of objects
        Function arguments.

    seed : list of floats or a :py:mod:`numpy` array
        Values to fill into arg during deep-copy.

    Returns
    -------
    arg
        A deep copy of arg filled with seed.

    """
    if not isinstance(seed, FillHelper):
        seed = FillHelper(seed)
    if isinstance(arg, list):
        return [fill(f, seed) for f in arg]
    elif isinstance(arg, tuple):
        return tuple(fill(f, seed) for f in arg)
    elif isinstance(arg, dict):
        return {f: fill(v, seed) for f, v in arg.items()}
    elif isgeneric(arg):
        return next(seed)
    elif hasattr(arg, 'flat'):
        arg.flat[:] = seed.get(arg.size)
        return arg


def dargs(args, seed=1):
    zargs = dzeros(args)
    if seed == 1:
        seed = [0] * nvars(args)
        seed[0] = 1
    dargs = fill(zargs, seed)
    return dargs


def createFullGradients(args):
    N = nvars(args)
    seeds = []
    for i in range(N):
        seed = [0] * N
        seed[i] = 1
        seeds.append(seed)
    return seeds


transformOpts = {}


def DiffFor(function, *args, seed=1, active=[], f_kw=None,
            timings=True, verbose=0, dump=0, dumpdir='dump', **opts):
    """Differentiate ``function`` and compute first-order derivatives
    evaluated at ``*args`` and ``**f_kw``, w.r.t. all floats in
    ``args``, possibly restricted by ``àctive``.

    Differentiate function ``function(*args)`` with forward mode AD to
    produce ``adfun``. This function is the main entry point to start
    the differentiation process. This function basically does the
    following:

       - Differentiate ``function`` with :py:func:`.DiffFunction` alias
         :py:func:`.D`

       - Create one set of derivative arguments ``dx = dzeros(args)``
         using :py:func:`.dzeros`

       - For each seeddir in seed, initialize ``dx`` with seeddir
         using :py:func:`.fill` and call ``adfun`` with ``dx`` and
         ``args`` appropriately.

    The result is a tuple of the list of the derivative results thus
    produced, and the function result.

    Although PyADi supports almost the full set of Python language
    features including keyword arguments, lambda functions, etc. the
    ``function`` given here must adhere to some restrictions:

      - ``function`` must be a regular Python function, defined with
         ``def``, not a lambda expression.

      - This function only processes the positional arguments ``args``
        and considers all keyword arguments as options to the process,
        additional keyword arguments can be passed using ``f_kw``.

      - ``function`` can have parameter default values,

      - ``function`` can be a local function returned by whatever
        other function. This setup processs will not be
        differentiated.

      - ``function`` can also have a decorator, which will be
        differentiated.

    It may in some cases be required, and it is no problem, to create
    additional toplevel functions that can be given to this function,
    for example to wrap a lambda expression.

    It should not be required to build extra toplevel functions to
    inject global variables into the scope, since ``function`` can be
    a local function already. It will have access to the parent scopes
    as usual, but the values in it are treated as global values with
    zero derivative.

    However, when a function returning a function is called within
    ``function``, then this entire process, including possible calls
    to the result later, will be differentiated.

    Parameters
    ----------

    function : function
        Function to differentiate. Must be a regular function, defined
        with ``def``, can be a local function.

    args : list
        Function arguments. ``function`` will be differentiated with
        respect to all arguments or to those listed by ``active``.

    active : list or str
        Active arguments, like [0,1], ['x', 'y'], or a comma-separated
        string like 'x,y'. The empty list or string means all
        arguments. What actually happens is that a local function of
        only the active ``args``, calling ``function``, is generated
        by :py:func:`.mkActArgFunction` and that is differentiated
        instead.

    seed : 1 or list
        Seed, derivative directions. When seed == 1, all derivative
        directions are computed. When seed is a list, then each entry
        must be a list or array of the same size as the total length
        of the active arguments. The function :py:func:`.nvars` can
        compute that value.

    f_kw : dict
        Further keyword arguments that will be passed to ``function``
        as ``**f_kw``. This wraps ``function`` with
        :py:func:`.mkKwFunction`.

    opts : dict
        Further options ``opts`` including also verbose, dump and
        dumpdir are stored in a global variable
        :py:data:`.transformOpts`. These global options are added to
        the options of :py:func:`.doDiffFunction` by each call to
        :py:func:`.D` in the subsequent process.

    Returns
    -------
    tuple
        A tuple of the derivative and the function result. The
        derivative is a list with as many entries as there were seed
        directions.

    """
    global transformOpts
    transformOpts = opts | dict(timings=timings, verbose=verbose, dump=dump, dumpdir=dumpdir)

    if dump > 0 and dumpdir != '.':
        global dumpDir
        dumpDir = dumpdir
        if not os.path.exists(dumpDir):
            print(f'mkdir {dumpDir}')
            os.makedirs(dumpDir)

    jacobian = opts.get('jacobian', True)

    if f_kw is not None:
        assert isinstance(f_kw, dict)
        function = mkKwFunction(function, f_kw)

    if len(active) > 0:
        inds = parinds(function, active)
        function, args = mkActArgFunction(function, args, inds)

    if timings:
        with Timer(function.__qualname__, 'run', verbose=verbose-1) as t:
            result = function(*args)

        with Timer(function.__qualname__, 'diff', verbose=verbose-1) as t:
            adfunOrig = D(function, **opts)

        def TimeIt(*args, **kw):
            with Timer(function.__qualname__, 'adrun', verbose=verbose) as t:
                return adfunOrig(*args, **kw)
        adfun = TimeIt

    else:

        adfun = D(function, **opts)

    if 'dx' in opts:
        dargs = dx
        dresult, result = adfun(*zip(fill(dargs, s), args))
    else:
        if isgeneric(seed) and seed == 1:
            seed = createFullGradients(args)
        elif isinstance(seed, list):
            pass
        else:
            raise ValueError()

        dargs = dzeros(args)
        dresult = []
        for s in seed:
            dresult.append(adfun(*zip(fill(dargs, s), args)))

        result = dresult[0][1] if len(dresult) else None
        dresult = [d for d, r in dresult]

    return dresult, result


def Diff(active='all', **opts):
    def _pyadi_diff(function):

        adc = {'f': None}

        def inner(*args, **kw):
            if 'mode' in kw and kw['mode'] == 'f':
                result = function(*args)
            else:
                result = function(*args)

                if adc['f'] is None:
                    (adfun, actind) = difffunction(function, active=active, **opts)
                    adc['f'] = (adfun, actind)
                else:
                    (adfun, actind) = adc['f']

                if 'dx' in kw:
                    dargs = dx
                else:
                    dargs = createGradients(args, actind)

                (dresult, result) = adfun(dargs, args)
                return (dresult, result)

        return inner
    return _pyadi_diff


def DiffFD(f, *args, active=[], seed=1, h=1e-8, f_kw={}, **opts):
    """Evaluate derivatves using central finite differences.

    The function f is called two times for each derivative direction
    provided by seed, to evaluate a central finite difference with
    step size h. The function f is called once more to compute the
    original result.

    Parameters
    ----------
    f : function
        Function to differentiate.

    args : list
        Function arguments.

    h  : float
        Step size, default 1e-8

    active : list or str
        Active arguments, like [0,1], ['x', 'y'], or a comma-separated
        string like 'x,y'. The empty list or string means all
        arguments. What actually happens is that a local function of
        only the active ``args``, calling ``function``, is generated
        by :py:func:`.mkActArgFunction` and that is differentiated
        instead.

    seed : 1 or list
        Seed, derivative directions. When seed == 1, all derivative
        directions are computed. When seed is a list, then each entry
        must be a list or array of the same size as the total length
        of the active arguments. The function :py:func:`.nvars` can
        compute that value.

    f_kw : dict
        Further keyword arguments that will be passed to ``f`` as
        ``**f_kw``.

    opts : dict
        Options, not used.

    Returns
    -------
    tuple
        A tuple of the derivative and the function result. The
        derivative is a list with as many entries as there were seed
        directions.

    """

    if len(active) == 0:
        func = f
    else:
        inds = parinds(f, active)
        func, args = mkActArgFunction(f, args, inds)

    v = np.array(list(varv(args)))
    dargs = dzeros(args)
    N = v.size
    h2 = h*2
    r = func(*args, **f_kw)

    def dirder(func, seed):
        #print('FDD', v, seed)
        r1 = func(*fill(dargs, v + h * seed), **f_kw)
        r2 = func(*fill(dargs, v - h * seed), **f_kw)
        #print('FDD', r1, r2)
        rv1 = np.array(varv(r1))
        rv2 = np.array(varv(r2))
        der = (rv1 - rv2)/h2
        return fill(dzeros(r), der)

    # print('seed', seed)
    if isgeneric(seed) and seed == 1:
        dres = []
        for i in range(N):
            seed = np.zeros(N)
            seed[i] = 1
            dres.append(dirder(func, seed))
    elif isinstance(seed, list):
        dres = [ dirder(func, np.array(seeddir)) for seeddir in seed ]
    else:
        raise ValuseError()
    return dres, r


def DiffFDNP(f, *args, active=[0], seed=1, h=1e-8, f_kw={}, **opts):
    """An optimized version of :py:func:`DiffFD` with some
    restrictions:

      - there can be only one active argument, considering
        opts['active'].

      - the only active argument, the seeds, and the function result
        must all by :py:mod:`numpy` arrays.

    Parameters
    ----------
    f : function
        Function to differentiate.

    args : list
        Function arguments.

    h  : float
        Step size.

    active : list or str
        Active arguments, like [0,1], ['x', 'y'], or a comma-separated
        string like 'x,y'. What actually happens is that a local function of
        only the active ``args``, calling ``function``, is generated
        by :py:func:`.mkActArgFunction` and that is differentiated
        instead.

    seed : 1 or list
        Seed, derivative directions. When seed == 1, all derivative
        directions are computed. When seed is a list, then each entry
        must be an array of the same size as the active argument.

    f_kw : dict
        Further keyword arguments that will be passed to ``f`` as
        ``**f_kw``.

    opts : dict
        Options, of which this funciton uses:

    Returns
    -------
    tuple
        A tuple of the derivative and the function result. The
        derivative is a list with as many entries as there were seed
        directions.

    """

    if len(active) == 0:
        func = f
    else:
        inds = parinds(f, active)
        func, args = mkActArgFunction(f, args, inds)

    assert(len(args) == 1)

    v = args[0]
    N = v.size
    sh = v.shape
    h2 = h*2
    r = func(*args, **f_kw)

    if isgeneric(seed) and seed == 1:
        seed = np.eye(N)

    if isinstance(seed, list):
        getcol = lambda i: seed[i]
        ndd = len(seed)
    else:
        getcol = lambda i: seed[:,i]
        n, ndd = seed.shape

    Jac = np.zeros((r.size, ndd))
    for i in range(ndd):
        v1 = v + h * getcol(i).reshape(sh)
        v2 = v - h * getcol(i).reshape(sh)

        r1 = func(v1, **f_kw)
        r2 = func(v2, **f_kw)
        Jac[:,i] = (r1.flat[:] - r2.flat[:])/h2

    return Jac, r

# (c) 2023 AI & IT UG
# Author: Johannes Willkomm jwillkomm@ai-and-it.de
