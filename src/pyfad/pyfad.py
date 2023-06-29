from astunparse import Unparser
import sys
import os
import inspect
import json
import shutil
from io import StringIO
import tempfile
from itertools import chain

from astunparse import loadast, unparse2j, unparse
from astunparse.astnode import ASTNode, BinOp, Constant, Name, isgeneric, fields

from .astvisitor import canonicalize, resolvetmpvars, normalize, filterLastFunction, infoSignature, filterFunctions, py, getmodule, getast
from .astvisitor import ASTVisitorID, mkTmp
from .nodes import *

from .timer import Timer

from . import astvisitor

from . import rules

Debug = False

dpref_ = 'd_'

def setprefix(diff, tmp, common=''):
    global dpref_
    dpref_ = common + diff
    astvisitor.setprefix(common + tmp)


def czip(a, b):
    return chain(*zip(a, b))


class NoRule(BaseException):
    pass


def nodiff(tree):
    return tree._class == "Constant"


def isdiff(tree):
    return not nodiff(tree)


class ASTVisitorFMAD(ASTVisitorID):

    active_objects = ['self', 'dself', 'dt']
    active_fields = ['acc', 'vel', 'pos', 'axis', 'x', 'y', 'z']
    active_methods = ['equations']

    def ddispatch(self, tree):
        if isinstance(tree, list):
            return [self.ddispatch(t) for t in tree]
        elif isgeneric(tree):
            return tree
        cname = tree._class
        meth = getattr(self, "_D"+cname, None)
        print('ddispatch?', cname, vars(tree))
        if meth:
            print('Found method', cname)
            return meth(tree)
        else:
            print('start dispatch', vars(tree).keys())
            print('start dispatch', dir(tree))
            res = ASTNode()
            for name in vars(tree).keys():
                delem = self.ddispatch(getattr(tree, name))
                print(f'DDispatch {name} => {repr(delem)}')
                setattr(res, name, delem)
            return res

    nodiffFunctions = ["print", "open"]
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
        for item in body:
            if item._class == "Assign":
                if item.value._class == "BinOp" and item.value.op == "**":
                    self.tmpval = mkTmp('s')
                    nbody += [Assign(self.tmpval, self.mkOpPartialC("**", None, None, item.value.left, item.value.right))]
                nbody += [self.ddispatch(item.clone())]
                if item.value._class == "BinOp" and item.value.op == "**":
                    nbody += [Assign(item.targets, self.tmpval)]
                    self.tmpval = None
                    continue
                if item.value._class not in ['Call', 'List', 'ListComp']:
                    nbody += [self.dispatch(item)]
            elif item._class == "AugAssign":
                nbody += [self.ddispatch(item.clone())]
                nbody += [self.dispatch(item)]
            elif self.isnodiffExpr(item):
                nbody += [self.dispatch(item)]
            else:
                nbody += [self.ddispatch(item)]
        return nbody

    def _FunctionDef(self, t):
        if t.name in self.active_methods:
            t = t.clone()
            print(f'Catch Active FunctionDef {t.name} {vars(t)}')
            t.name = dpref_ + t.name
            t.args = self.ddispatch(t.args)
            t.body = self.diffStmtList(t.body)
            t.decorator_list = []
        else:
            t.args = self.dispatch(t.args)
            t.body = self.dispatch(t.body)
        return t

    def _DIfExp(self, node):
        node.body = self.ddispatch(node.body)
        node.orelse = self.ddispatch(node.orelse)
        return node

    def _DSubscript(self, node):
        node.value = self.ddispatch(node.value)
        return node

    def diffUnlessIsTupleDiff(self, t, src=None):
        if t._class == "Call" or t._class == "List" or t._class == "ListComp":
            res = self.ddispatch(t.clone())
            if src and src._class == 'Call':
                if t._class == "List":
                    res = res.elts[0].value
            return res
        else:
            return Tuple([self.ddispatch(t.clone()),t])

    def _DList(self, node):
        dargs = [self.diffUnlessIsTupleDiff(t) for t in node.elts]
        return List([Starred(Call('zip', dargs))])

    def _DDict(self, node):
        node.values = self.ddispatch(node.values)
        return node

    def _DDictComp(self, node):
        node.value = self.ddispatch(node.value)
        return node

    def _DListComp(self, node):
        node.elt = self.diffUnlessIsTupleDiff(node.elt)
        node.generators = self.ddispatch(node.generators)
        return Call('zip', [Starred(node)])

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
        tnode = Tuple([self.ddispatch(node.target), node.target])
        if node.iter._class == "Call":
            itnode = Call('zip', [Starred(self.ddispatch(node.iter))])
        else:
            itnode = Call('zip', [self.ddispatch(node.iter), self.dispatch(node.iter)])
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
            if t.arg in self.active_objects:
                tr1 = self.ddispatch(t)
                dargs += [tr1]
        node.args = list(chain(*zip(dargs, curargs)))
        ddefs = self.ddispatch(node.defaults)
        node.defaults = list(chain(*zip(ddefs, node.defaults)))
#        node.args = dargs + curargs
        return node

    nonder_builtins = ['len']

    def _DCall(self, t):
        print(f'Diff Call {t.func} {vars(t)}')
        t = t.clone()
        dcall = Call(Name('D'))
        dcall.args = [t.func]

        res = Call(dcall)
        curargs = t.args

        if t.func._class == "Attribute":
            print('ATTR', t.func.attr)
            attrstr = unparse(t.func.value).strip()
            if attrstr not in self.imports:
                curargs = [t.func.value] + curargs

        dargs = [self.diffUnlessIsTupleDiff(a, t) for a in curargs]
        res.args = dargs
        res.keywords = self.ddispatch(t.keywords) + self.dispatch(t.keywords)
        return res

    def _Darg(self, t):
        if t.arg in self.active_objects:
            t = t.clone()
            t.arg = dpref_ + t.arg
            print('   * active arg', t.arg)
        return t

    def _Dkeyword(self, t):
        t = t.clone()
        t.arg = dpref_ + t.arg
        t.value = self.ddispatch(t.value)
        return t

    def _DAssign(self, t):
        if t.value._class == 'Call' or t.value._class == "List" or t.value._class == "ListComp":
            t.targets = [Tuple(self.ddispatch(t.targets) + self.dispatch(t.targets))]
        else:
            t.targets = self.ddispatch(t.targets)
        isList = t.value._class == "List"
        t.value = self.ddispatch(t.value)
        if isList:
            assert t.value.elts[0]._class == "Starred"
            t.value = t.value.elts[0].value
        return t

    def _DName(self, t):
        print(f'Diff Name {t.id}')
        t = t.clone()
        t.id = dpref_ + t.id
        return t

    def _DAttribute(self, t):
        print(f'Diff Attribute {t.attr} of {vars(t.value)} {self.imports}')
        t.value = self.ddispatch(t.value.clone())
        return t

    def _DConstant(self, t):
        t = t.clone()
        t.value = 0
        return t

    def mkOpPartialC(self, op, r, dx, x, y):
        if op == '**':
            if r is None:
                t = BinOp(op, x, y)
            else:
                t = r
        return t

    def mkOpPartialL(self, op, r, dx, x, y):
        if op == '/':
            t = BinOp('/', dx, y)
        elif op == '%':
            t = dx
        elif op == '**':
            quot = BinOp('/', y, x)
            t = BinOp('*', quot, dx)
        return t

    def mkOpPartialR(self, op, r, dy, x, y):
        if op == '/':
            sq = BinOp('**', y, Constant(2))
            right_ = BinOp('*', x, dy)
            t = UnaryOp('-', BinOp('/', right_, sq))
        elif op == '%':
            quot = BinOp('/', x, y)
            t = BinOp('*', BinOp('*', Call('math.floor', [quot]), Constant(-1)), dy)
        elif op == '**':
            t = BinOp('*', Call('log', [x]), dy)
        return t

    def _DBinOp(self, t):
        print(f'Diff BinOp {t} left {vars(t.left)}')
        if nodiff(t.left) and nodiff(t.right):
            return Constant(0.0)

        if nodiff(t.left):
            left = self.dispatch(t.left)
        else:
            left = self.ddispatch(t.left.clone())
        if nodiff(t.right):
            right = self.dispatch(t.right)
        else:
            right = self.ddispatch(t.right.clone())

        if t.op == '*':

            if isdiff(t.left) and isdiff(t.right):
                left_ = BinOp('*', left, t.right)
                right_ = BinOp('*', t.left, right)
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

        elif t.op == '%':
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
            fact = self.mkOpPartialC('**', self.tmpval, None, t.left, t.right)

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

            t = BinOp('*', fact, term)

        elif t.op == '+' or t.op == '-':
            t.left = left
            t.right = right

        else:
            t.left = self.dispatch(t.left)
            t.right = self.dispatch(t.right)
        return t

    def _DReturn(self, t):
        if t.value._class == "Call":
            t.value = self.ddispatch(t.value)
        else:
            t.value = Tuple([self.ddispatch(t.value.clone()), self.dispatch(t.value)])
        return t


def diff2pys(intree, visitor, *kw):
#   print('intree', unparse2j(intree, indent=1), file=open('intree.json', 'w'))
    intree = canonicalize(intree)
#    intree = resolvetmpvars(intree)
#    print('canon', unparse2j(intree, indent=1), file=open('canon.json', 'w'))
#    print('canon', unparse(intree), file=open('canon.py', 'w'))
    print('canon', unparse(intree))
    intree = normalize(intree.clone())
#    print('canon', unparse2j(intree, indent=1), file=open('norm.json', 'w'))
#    print('canon', unparse(intree), file=open('norm.py', 'w'))
#    print('canon', unparse(intree))
    outtree = visitor(intree)
#    print('outtree', unparse2j(outtree, indent=1), file=open('outtree.json', 'w'))
#    outtree = resolvetmpvars(outtree)
    print('outtree', unparse2j(outtree, indent=1), file=open('outtree2.json', 'w'))
    return outtree


def differentiate(intree, activef=None, active=None, modules=None, filter=False, prefix=None, **kw):
    fmadtrans = ASTVisitorFMAD()

    if prefix:
        while len(prefix) < 3:
            prefix.append('')
        setprefix(*prefix)

    fmadtrans.imports = modules
    print('imports', fmadtrans.imports)
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
    fmadtrans.active_objects = ['self', 'dself', 'dt'] + fmadtrans.active_fields

    dtree = diff2pys(intree, fmadtrans)
    return dtree


def diff2py(fname):
    with open(fname, "r") as pyfile:
        source = pyfile.read()
    return unparse(diff2pys(source, fname))


def diff2pys2s(source, fname):
    return unparse(differentiate(loadast(source)))


def roundtrip2JID(fname):
    with open(fname, "r") as pyfile:
        source = pyfile.read()
        return roundtrip2JIDs(source, fname)


def execompile(source, fglobals={}, flocals={}, imports=['math', 'sys', 'os', {'pyfad': 'D'}], vars=['x'], **kw):

    importstr = '\n'.join([f'import {name}' if isinstance(name, str) else ('\n'.join([f'from {k} import {v}' for k, v in name.items()])) for name in imports])
    collectstr = '\n'.join([f'data["{name}"] = {name}' for name in vars])

#    dsrc = f"{importstr}\n{source}\n{collectstr}"
    dsrc = f"{source}\n{collectstr}"
    print(f"{source}")
    sfname = ""
    if Debug or True:
        tmpsdir = tempfile.mkdtemp(prefix='pyfad_')
        sfname = f'{tmpsdir}/diff.py'
        with open(sfname, 'w') as f:
            f.write(dsrc)
    try:
        res = compile(dsrc, sfname, "exec")
    except SyntaxError as ex:

        print('Failed to compile python code', ex)
        print(dsrc)
        if not sfname:
            tmpsdir = tempfile.mkdtemp(prefix='pyfad_')
            sfname_ = f'{tmpsdir}/diff.py'
            with open(sfname_, 'w') as f:
                f.write(dsrc)
        else:
            sfname_ = sfname
        res = compile(dsrc, sfname_, "exec")
        if not sfname:
            shutil.rmtree(tmpsdir)

    gvars = {'data': {}}
    exec(res, gvars | globals() | fglobals, flocals)

    result = {name: gvars["data"][name] for name in vars}

    return result


def Dpy(func, active=[]):
    csrc, imports, modules = getast(func)
    dtree = differentiate(csrc, activef=func.__name__, active=active, modules=modules)
    return dtree


def difffunction(func, active=[]):
    dsrc = Dpy(func, active)
    try:
        fkey = dpref_ + func.__name__
        # globals = func.__globals__ if not isinstance(func, type) else func.__init__.__globals__
        dfunc = execompile(dsrc, vars=[fkey], fglobals=func.__globals__)
        dfunc = dfunc[fkey]
    except BaseException as ex:
        print(unparse2j(dsrc, indent=1), file=open('d_failed.json', 'w'))
        print(unparse(dsrc), file=open('d_failed.py', 'w'))
        print(f"""Failed to load diff code, exception:
{ex}
Source:
{py(func)}
""")
        raise ex
    return (dfunc, active)


def run():
    fname = 'pyphy/parser.py'
    with open(fname) as f:
        code = f.read()
        output = sys.stdout
        tree = compile(code, fname, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
        c1 = ASTFragment(tree)
        Unparser(tree, output)

        src = inspect.getsource(Test.energy).strip()
        print(f'prop: "\n{src}"')
        roundtrip2Js(src, 'test_py')


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


def fid(func, active):
    mod, modfile = getmodule(func)
    if modfile is None:
        modfile = mod
    fid = f'{func.__qualname__}:{modfile}:{repr(active)}'
#    print('FID', func, fid)
    return fid


def is_instance_userdefined_and_newclass(inst):
    cls = inst.__class__
    if hasattr(cls, '__class__'):
        return ('__dict__' in dir(cls) or hasattr(cls, '__slots__'))
    return False


def isbuiltin(func):
    mod, modfile = getmodule(func)
    res = modfile is None
#    print('isbuiltin', func, res)
    return res


def setrule(func, adfunc):
    id = 'D_' + rid(func)
    print(f'set AD rule for {func.__name__}, key {id}')
    setattr(rules, id, adfunc)
    rules.dict[id] = adfunc


def delrule(func):
    id = 'D_' + rid(func)
    print(f'clear AD rule for {func.__name__}, key {id}')
    if id in rules.dict:
        del rules.dict[id]
    else:
        rules.hidden[id] = getattr(rules, id)
    delattr(rules, id)


def restorerule(func):
    id = 'D_' + rid(func)
    print(f'restore AD rule for {func.__name__}, key {id}')
    if id in rules.hidden:
        setattr(rules, id, rules.hidden[id])
        del rules.hidden[id]


def getrules():
    return rules.dict


def rid(func):
    mod, _ = getmodule(func)
    fid = f'{func.__qualname__}_{mod}'.replace('.', '_')
#    print('Rule ID', func, fid)
    return fid


def getsig(f):
    x = inspect.signature(f)
    x = [f for f in x.parameters]
    return x


def varspec(x):
    if isinstance(x, str):
        x = x.split(',')
    return x


adc = {}


def clear(search=None):
    global adc
    if search is None:
        adc = {}
        astvisitor.astcache = {}
    elif isinstance(search, str):
        for k in adc:
            if search in k:
                del adc[k]
        for k in astvisitor.astcache:
            if search in k:
                del astvisitor.astcache[k]
    else:
        del adc[fid(search)]


def runRule(adfun, function, args):
    res = function(*args[1::2])
    return (adfun(res, *args), res)


def DiffFunction(function, **opts):

    id = 'D_' + rid(function)
    adfun = getattr(rules, id, None)
    _class = None

    if isbuiltin(function) and adfun is None:
        fname = function.__name__
        msg = f'No rule for buitin {fname}, function {id} not found'
        raise (NoRule(msg))

    if adfun is None:

        if isinstance(function, type):
            if not isbuiltin(function.__init__):
                _class = function
                function = function.__init__
            else:
                def initDObj():
                    do, o = function(), function()
                    do = dzeros(do)
                    print('dobj', do.velocity)
                    return do, o
                adfun = lambda: initDObj()
                return adfun

        # Try source diff
        active = opts.get('active', [])
#        print('DDD', active)
        if _class:
            adfun = getattr(_class, id, None)

        if adfun is not None:
            print(f'Diff function {function.__name__} found as class attr')
        else:
            findex = fid(function, active)
            if findex in adc:
                print(f'Found diff function {function.__name__}')
                (adfun, actind) = adc[findex]
            else:
                print(f'Diff function {function.__name__}')
                with Timer(function.__qualname__, 'diff') as t:
                    (adfun, actind) = difffunction(function, active=active)
                adc[findex] = (adfun, actind)
                print(f'Diff function {function.__name__} cached => {findex}')

            if _class:
                dfname = f'd_{function.__name__}'
                setattr(_class, dfname, adfun)
                print(f'Diff function {function.__name__} saved as attr {dfname} in type {_class.__qualname__}')

        adfunSrc = adfun
        def TimeIt(*args, **kw):
            with Timer(function.__qualname__, 'adrun') as t:
                return adfunSrc(*args, **kw)

        adfun = TimeIt
        adfun.issource = True

    else:

        adfun.issource = False
        adfunOrig = adfun

        def inner(*args):
            return runRule(adfunOrig, function, args)
        adfun = inner

    def inner2(*args, **kw):
        # assert len(args) == 0 or len(list(args[0])) == 2
        args = chain(*args)
        return adfun(*args, **kw)

    return inner2


D = DiffFunction


def nvars(args):
    if isinstance(args, list) or isinstance(args, tuple):
        return sum([nvars(f) for f in args])
    elif isinstance(args, dict):
        return sum([nvars(v) for f, v in args.items()])
    elif isgeneric(args):
        return 1


def varv(args):
    if isinstance(args, list) or isinstance(args, tuple):
        return chain(*[varv(f) for f in args])
    elif isinstance(args, dict):
        return chain(*[varv(v) for f, v in args.items()])
    elif isgeneric(args):
        return [args]


def dzeros(args):
    if isinstance(args, list):
        return [dzeros(f) for f in args]
    elif isinstance(args, tuple):
        return tuple([dzeros(f) for f in args])
    elif isinstance(args, dict):
        return {f: dzeros(v) for f, v in args.items()}
    elif isgeneric(args):
        return 0.0
    elif isinstance(args, object):
        # we assume the object is already allocated
        for a in fields(args, True):
            setattr(args, a, dzeros(getattr(args, a)))
        return args


class FillHelper:
    def __init__(self, seed):
        self.seed = seed
        self.offs = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.offs < len(self.seed):
            r = self.seed[self.offs]
            self.offs += 1
            return r
        else:
            raise StopIteration


def fill(arg, seed):
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
    zargs = dzeros(args)
    for i in range(N):
        seed = [0] * N
        seed[i] = 1
        dargs = fill(zargs, seed)
        seeds.append(dargs)
    return seeds


def DiffFor(function, *args, **opts):
    with Timer(function.__qualname__, 'run') as t:
        result = function(*args)

    adfun = D(function, **opts)

    seed = opts.get('seed', 1)

    if 'dx' in opts:
        dargs = dx
    else:
        if seed == 1:
            dargsList = createFullGradients(args)
            dresult = [adfun(*zip(dargs, args)) for dargs in dargsList]
            result = dresult[0][1]
            dresult = [d for d, r in dresult]
        elif isinstance(arg, list):
            dargs = fill(dzeros(args), seed)
            (dresult, result) = adfun(*czip(dargs, args))

    return (dresult, result)


def Diff(active='all'):
    def _pyfad_diff(function):

        adc = {'f': None}

        def inner(*args, **kw):
            if 'mode' in kw and kw['mode'] == 'f':
                result = function(*args)
            else:
                result = function(*args)

                if adc['f'] is None:
                    (adfun, actind) = difffunction(function, active=active)
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
    return _pyfad_diff


def DiffFD(f, *args, **opts):

    seed = opts.get('seed', 1)
    active = opts.get('active', [])
    h = opts.get('h', 1e-8)

    if len(active) == 0:
        func = f
    else:
        active = varspec(active)
        sig = getsig(f)
        inds = [sig.index(a) for a in sig if a in active]
        fullargs = [v for v in args]

        def inner(*aargs):
            for i, k in enumerate(inds):
                fullargs[k] = aargs[i]
            return f(*fullargs)
        func = inner
        args = [args[i] for i in inds]

    N = nvars(args)
    v = list(varv(args))
    h2 = h*2
    r = func(*args)

    def dirder(func, args, seed):
        v1 = [v[i] + h * seed[i] for i in range(N)]
        v2 = [v[i] - h * seed[i] for i in range(N)]
        r1 = func(*fill(args, v1))
        r2 = func(*fill(args, v2))
        rv1 = varv(r1)
        rv2 = varv(r2)
        der = ([(rv1[i] - rv2[i])/h2 for i in range(len(rv1))])
        return fill(r, der)

    if seed == 1:
        dres = []
        for i in range(N):
            seed = [0] * N
            seed[i] = 1
            dres.append(dirder(func, args, seed))
    else:
        dres = dirder(func, args, seed)
    return dres, r
