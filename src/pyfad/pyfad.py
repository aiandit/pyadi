from astunparse import Unparser
import sys, os, inspect, json, shutil
from io import StringIO
import tempfile
from itertools import chain

from astunparse import loadast, unparse2j, unparse
from astunparse.astnode import ASTNode, BinOp, Constant, Name, isgeneric
from .astvisitor import ASTVisitorID, canonicalize, resolvetmpvars, normalize, Assign, List, Tuple, py

from . import rules

class NoRule(BaseException):
    pass

class Call(ASTNode):
    def __init__(self, func):
        self._class = "Call"
        self.func = func
        self.args = []
        self.keywords = []

class Keyword(ASTNode):
    def __init__(self, arg, value):
        self._class = "Keyword"
        self.arg = arg
        self.value = value

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
                    if item.value._class not in ['Call']:
                        nbody += [ self.dispatch(item) ]
                elif item._class == "AugAssign":
                    nbody += [ self.ddispatch(item.clone()) ]
                    nbody += [ self.dispatch(item) ]
                elif item._class == "Return":
                    nbody += [self.ddispatch(item)]
                else:
                    nbody += [ self.dispatch(item) ]
            t.body = nbody
            t.decorator_list = []
        else:
            t.args = self.dispatch(t.args)
            t.body = self.dispatch(t.body)
        return t

    def _Darguments(self, node):
        assert(type(node.args) == type([]))
        dargs = []
        for t in node.args:
            if t.arg in self.active_objects:
                tr1 = self.ddispatch(t)
                dargs += [tr1]
        node.args = dargs + node.args
        return node

    def _Darg(self, t):
        if t.arg in self.active_objects:
            t = t.clone()
            t.arg = 'd_' + t.arg
            print('   * active arg', t.arg)
        return t

    def _DAssign(self, t):
        if t.value._class == 'Call':
            t.targets = [Tuple(self.ddispatch(t.targets) + self.dispatch(t.targets))]
        else:
            t.targets = self.ddispatch(t.targets)
        t.value = self.ddispatch(t.value)
        return t

    def _DCall(self, t):
        print(f'Diff Call {t.func} {vars(t)}')
        t = t.clone()
        dcall = Call(Name('D'))
        dcall.args = [t.func]

        res = Call(dcall)
        res.args = self.ddispatch([t.clone() for t in t.args]) + self.dispatch(t.args)
        res.keywords = self.ddispatch(t.keywords)
        return res

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
        print(f'Diff BinOp {t} left {vars(t.left)}')
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
        print(f'Catch BinOp {repr(t.op)}')
        return t

    def _DReturn(self, t):
        print('Diff Return')
        t.value = Tuple([self.ddispatch(t.value.clone()), self.dispatch(t.value)])
        return t


def diff2pys(intree, visitor, *kw):
    print('intree', unparse2j(intree, indent=1), file=open('intree.json', 'w'))
    intree = canonicalize(intree)
    intree = resolvetmpvars(intree.clone())
    print('canon', unparse2j(intree, indent=1), file=open('canon.json', 'w'))
    print('canon', unparse(intree), file=open('canon.py', 'w'))
    print('canon', unparse(intree))
    intree = normalize(intree.clone())
    print('canon', unparse2j(intree, indent=1), file=open('norm.json', 'w'))
    print('canon', unparse(intree), file=open('norm.py', 'w'))
    print('canon', unparse(intree))
    outtree = visitor(intree)
    print('outtree', unparse2j(outtree, indent=1), file=open('outtree.json', 'w'))
    return outtree

def differentiate(intree, activef=None, active=None, **kw):
    fmadtrans = ASTVisitorFMAD()
    fmadtrans.active_methods = activef if activef is not None else []
    fmadtrans.active_fields  = active if active is not None else []
    fmadtrans.active_objects = ['self', 'dself', 'dt', 'forces'] + active if active is not None else []
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

    importstr = '\n'.join([f'import {name}' if isinstance(name, str) else ('\n'.join([f'from {k} import {v}' for k,v in name.items()])) for name in imports])
    collectstr = '\n'.join([f'data["{name}"] = {name}' for name in vars])

#    dsrc = f"{importstr}\n{source}\n{collectstr}"
    dsrc = f"{source}\n{collectstr}"
    print(dsrc)
    tmpsdir = tempfile.mkdtemp(prefix='pyfad_')
    sfname = f'{tmpsdir}/diff.py'
    with open(sfname, 'w') as f:
        f.write(dsrc)
    res = compile(dsrc, sfname, "exec")

    gvars = {'data': {}}
    print('compiling with globals', (gvars|globals()|fglobals).keys())
    exec(res, gvars|globals()|fglobals, flocals)

    result = {name: gvars["data"][name] for name in vars}

#    shutil.rmtree(tmpsdir)
    return result


def Dpy(func, active=[]):
    csrc = py(func)
    dtree = differentiate(loadast(csrc), activef=[func.__name__, func.__qualname__], active=active)
    dsrc = unparse(dtree)
    return dsrc


def difffunction(func, active=[]):
    dsrc = Dpy(func, active)
    try:
        fkey = 'd_' + func.__name__
        dfunc = execompile(dsrc, vars=[fkey], fglobals=func.__globals__)
        dfunc = dfunc[fkey]
    except BaseException as ex:
        print(unparse2j(dtree, indent=1), file=open('d_failed.json', 'w'))
        print(dsrc, file=open('d_failed.py', 'w'))
        print(f"""Failed to load diff code, exception:
{ex}
Source:
{py(func)}
Result:
{dsrc}""")
        raise(ex)
    return (dfunc, active)


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

def fid(func,active):
    fmod = func.__module__
    modfile = getattr(sys.modules[fmod], '__file__', fmod)
    fid = f'{func.__qualname__}:{modfile}:{repr(active)}'
    print('FID', func, fid)
    return fid


def isbuiltin(func):
    fmod = func.__module__
    return getattr(sys.modules[fmod], '__file__', None) is None


def setrule(func, adfunc):
    id = 'D_' + rid(func)
    print(f'set AD rule for {func.__name__}, key {id}')
    setattr(rules, id, adfunc)


def clearrule(func):
    id = 'D_' + rid(func)
    print(f'clear AD rule for {func.__name__}, key {id}')
    delattr(rules, id)


def rid(func):
    fmod = func.__module__
    fid = f'{func.__qualname__}_{fmod}'
    print('Rule ID', func, fid)
    return fid


def varspec(f, x):
    if isinstance(x, str):
        if x == "all":
            x = inspect.signature(f)
            x = [f for f in x.parameters]
        else:
            x = x.split(',')
        return x
    else: return x


adc = {}
def clear(search=None):
    global adc
    if search is None:
        adc = {}
    else:
        for k in adc:
            if search in k:
                del adc[k]


def DiffFunction(function, opts={'active': 'all'}):

    if isbuiltin(function):
        id = 'D_' + rid(function)

        adfun = getattr(rules, id, None)
        if adfun is None:
            raise(NoRule(f'No rule for {function}, {id} not found in rules'))

    else:

        active = varspec(function, opts['active']) if 'active' in opts else []
        findex = fid(function,active)
        if findex in adc:
            print(f'Found diff function {function.__name__}')
            (adfun, actind) = adc[findex]
        else:
            print(f'Diff function {function.__name__}')
            (adfun, actind) = difffunction(function, active=active)
            adc[findex] = (adfun, actind)
            print(f'Diff function {function.__name__} cached => {findex}')

    return adfun

D = DiffFunction

def nvars(args):
    if isinstance(args, list) or isinstance(args, tuple):
        return sum([nvars(f) for f in args])
    elif isinstance(args, dict):
        return sum([nvars(v) for f,v in args.items()])
    elif isgeneric(args):
        return 1


def varv(args):
    if isinstance(args, list) or isinstance(args, tuple):
        return chain(*[varv(f) for f in args])
    elif isinstance(args, dict):
        return chain(*[varv(v) for f,v in args.items()])
    elif isgeneric(args):
        return [args]


def dzeros(args):
    if isinstance(args, list):
        return [dzeros(f) for f in args]
    elif isinstance(args, tuple):
        return tuple([dzeros(f) for f in args])
    elif isinstance(args, dict):
        return {f: dzeros(v) for f,v in args.items()}
    elif isgeneric(args):
        return 0.0


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
        return {f: fill(v, seed) for f,v in arg.items()}
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

def DiffFor(function, args, seed=1, opts={'active': 'all'}):
    result = function(*args)

    adfun = D(function, opts)

    if 'dx' in opts:
        dargs = dx
    else:
        if seed == 1:
            dargsList = createFullGradients(args)
            dresult = [adfun(*dargs, *args) for dargs in dargsList]
            result = dresult[0][1]
            dresult = [d for d,r in dresult]
        elif isinstance(arg, list):
            dargs = fill(dzeros(args), seed)
            (dresult, result) = adfun(*dargs, *args)

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


def DiffFD(f, *args, opts=None, active=[], seed=1):

    if opts is None:
        if isinstance(args[len(args)-1], dict):
            opts = args[len(args)-1]
            args = args[0:-1]
        else:
            opts = {}

    seed = opts['seed'] if 'seed' in opts else seed
    active = varspec(f, opts['active'] if 'active' in opts else active)
    h = opts['h'] if 'h' in opts else 1e-8

    if len(active) == 0:
        func = f
    else:
        sig = varspec(f, 'all')
        inds = [sig.index(a) for a in sig if a in active]
        fullargs = [v for v in args]
        def inner(*aargs):
            for i,k in enumerate(inds):
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


@Diff(active=['x', 'y'])
def demof1(x,y):
    r = x*y

@Diff(active=['x', 'z'])
def demof2(x,y,z):
    r = x*y*z

def rundifffunc():
    x, y, z = 3,6,2
    (a, b) = demof2(x, y, z)
    print(a,b)
    (a, b) = demof2(x, y, z, dx=[1,2,3])
    print(a,b)

class Flywheel:
    pos = 0
    vel = 0
    acc = 0
    def equations(self, dt):
        self.pos += self.vel * dt
        self.vel += self.acc * dt

if __name__ == "__main__":
    rundifffunc()
    # rundiffmethod()
    # testdir()
#run()
