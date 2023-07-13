from astunparse import Unparser
import sys, os, inspect, json
from io import StringIO
import time
import random

import astunparse
from astunparse import loadastpy, unparse, unparse2j
from astunparse.astnode import fields
from .nodes import *

tpref_ = 't_'

def setprefix(t):
    global tpref_
    tpref_ = t


tmpseen = {}
class TmpVar(Name):
    def __init__(self, kind='t'):
        super().__init__(mkTmpName(kind))

def mkTmpName(kind='t'):
    for i in range(3):
        id = random.random()
        if id not in tmpseen:
            break
    short = f'{tpref_}{kind}{len(tmpseen):d}'
    tmpseen[id] = short
    return short

def mkTmp(kind='t'):
    return Name(mkTmpName(kind))

class NotFound(BaseException):
    pass

class NoSource(BaseException):
    pass


def fname(func):
    fname = getattr(func, '__qualname__', getattr(func, '__name__', None))
    return fname

def fqname(func):
    mod, _ = getmodule(func)
    fname = getattr(func, '__qualname__', getattr(func, '__name__', None))
    return f'{mod}.{fname}'


def rid(func):
    fname = fqname(func)
    #print(f'rid(func) = {func} {fname} {dir(func)}')
    fid = fname.replace('.', '_')
    return fid


def getmodule(func):
    #print('getmodule', func, type(func))
    mod = getattr(func, '__module__', None)
    if mod is None:
        mod = func.__class__.__module__
    modfile = getattr(sys.modules[mod], '__file__', None)
    #print('getmodule', func, mod, modfile)
    return mod, modfile


def isbuiltin(func):
    mod, modfile = getmodule(func)
    res = modfile is None
    return res


modastcache = {}
def getmoddict(mod, **opts):
    modfile = getattr(sys.modules.get(mod, {}), '__file__', None)
    if modfile is None or modfile.endswith('.so'):
        if opts.get('verbose', 0):
            print(f'No source for module {mod}')
        raise NoSource(f'No source for module {mod}')
    if mod in modastcache:
        centry = modastcache[mod]
        tree, imports, modules = centry["data"]
        moddict = centry["dict"]
    else:
        t0 = time.time()
        with open(modfile) as f:
            csrc = f.read()
        tree = loadastpy(csrc)
        imports, modules = ASTVisitorImports()(tree)
        moddict = ASTVisitorDict()(tree)
        modastcache[mod] = {'name': mod, "file": modfile, "data": (tree, imports, modules), "dict": moddict}
        resolveImports(mod, modfile, moddict, imports, modules, **opts)
        if opts.get('verbose', 0):
            print(f'Load and parse module {mod} source from {modfile}')
        t1 = time.time()
        if opts.get('verbose', 0):
            print(f'Load and parse module {mod} source from {modfile}: {1e3*(t1-t0):.1f} ms')
    return moddict, imports, modules


def resolveImports(mod, modfile, moddict, imports, modules, **opts):
    pkgs = mod.split('.')
    moduleImports = {}
    if opts.get('verbose', 0):
        print(f'Resolve imports for {mod}, {modfile}')
    # print(f'Resolve imports for {mod}, {modfile}:, imports={imports}, modules={modules}')
    # print(f'moddict={moddict.keys()}')
    for name in imports:
        impentry = imports[name]
        if isinstance(impentry, dict):
            assert len(impentry.keys()) == 1
            imod = list(impentry.keys())[0]
            if imod not in moduleImports:
                moduleImports[imod] = {}
            moduleImports[imod].update({ name:  impentry[imod] })
    if opts.get('verbose', 0) > 1:
        print(f'moduleImports={moduleImports}')
    for imod in moduleImports:
        modname, level = imod
        if modname is None:
            # these are all modules
            continue
        if level > 0:
            ilevel = level
            if modfile.endswith('__init__.py'):
                ilevel -= 1
            prepkgs = pkgs if ilevel <= 0 else pkgs[0:-ilevel]
            prepkgs += [modname]
            modname_ = '.'.join(prepkgs)
            if opts.get('verbose', 0) > 1:
                print(f'Get local import {modname}, {level}, {ilevel}, {pkgs} => {modname_}')
            modname = modname_
        try:
            impd, _, _ = getmoddict(modname)
            #print(f'Got moddict for {modname}: {impd.keys()}')
        except NoSource:
            continue
        imodimps = moduleImports[imod]
        for name, impname in imodimps.items():
            if name == "*":
                if opts.get('verbose', 0) > 0:
                    print(f'Import {impname} from import module {modname} as {name} into {mod}')
                moddict.update(impd)
            else:
                if impname in impd:
                    if opts.get('verbose', 0) > 0:
                        print(f'Import {impname} from module {modname} into {mod} as {name}')
                    moddict[name] = impd[impname]
                else:
                    if opts.get('verbose', 0) > 0:
                        print(f'Import {impname} from module {modname} into {mod} as {name} is likely a module.')
    return moddict


def getast(func, **kw):
    # ta0 = time.time()
    mod, modfile = getmodule(func)
    # print(f'Get SRC and AST: {func.__qualname__} in {mod} file {modfile}')
    if modfile is None:
        print(f'No source for {mod}.{fqname(func)}')
        raise(NoSource(f'No source for {mod}.{fqname(func)}'))

    moddict, imports, modules = getmoddict(mod, **kw)

    try:
        tree = moddict[fname(func)]
    except KeyError:
        print(f'No source for {mod}.{fname(func)}')
        raise(NoSource(f'No source for {mod}.{fname(func)} => {rid(func)}'))

    # ta1 = time.time()
    # print(f'Got AST of {mod}.{func.__name__}: {1e3*(ta1-ta0):.1f} ms')
    return tree, imports, modules


def py(func, info=False):
    tree, imports, modules = getast(func)
    src = unparse(tree).strip()
    if info:
        return src, imports, modules
    else:
        return src


class ASTVisitor:

    def __init__(self):
        # print('ASTVisitor()')
        pass

    def __call__(self, tree):
        # print('ASTVisitor.call()')
        self.result = self.dispatch(tree)
        return self.result

    def dispatch(self, tree):
        # print('ASTVisitor.dispatch()')
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
        # print(cname, vars(tree))
        if meth:
            return meth(tree)
        else:
            for name in vars(tree):
                setattr(tree, name, self.dispatch(getattr(tree, name)))
            return tree

    def _BinOp(self, t):
        # print(f'Catch BinOp {t.op}')
        t.left = self.dispatch(t.left)
        t.right = self.dispatch(t.right)
        return t


def isop(cn):
    return cn._class in ['BinOp', 'UnaryOp', 'BoolOp', 'AugAssign']

def iscall(cn):
    return cn._class in ['Call']

def iscanon(cn):
    return isop(cn) or iscall(cn)

class ASTCanonicalizer:
    def __init__(self):
        pass

    def __call__(self, tree):
        self._list = []
        self.active = False
        result = self.dispatch(tree)
        return result

    def edispatch(self, tree, val=None):
#        print('edisp', tree)
        if type(tree) == type([]):
            raise(BaseException('error'))
            res = list(map(self.edispatch, tree))
        elif isinstance(tree, ASTNode):
            tmpv = mkTmp('c')
            tmpas = Assign(tmpv, self.dispatch(tree.clone()) if val is None else val)
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

            if getattr(tree, 'body', None) is not None and tree._class != "Module" and tree._class != "IfExp" and tree._class != "Lambda":
                nbody = []
                for stmt in tree.body:
                    if stmt._class == "For":
                        self._list = []
                        self.active = True
                        stmt.iter = self.dispatch(stmt.iter)
                        nbody += self._list
                        self.active = False

                    if getattr(stmt, 'body', None) is None:
                        self._list = []
                        self.active = True
                        pstmt = self.dispatch(stmt)
                        nbody += self._list
                        nbody += [pstmt]
                        self.active = False
                    else:
                        nbody += [self.dispatch(stmt)]
                tree.body = nbody

                return tree
            elif tree._class == "DictComp" or tree._class == "ListComp":
                tree.generators = [self.dispatch(tree.generators)]
                return tree

            elif tree._class == "Subscript":
                if iscanon(tree.value):
                    (tl, tmpvar) = self.edispatch(tree.value)
                    tree.value = tmpvar
                return tree

            for k in fields(tree):
                setattr(tree, k, self.dispatch(getattr(tree, k)))

            if tree._class == "AugAssign" or tree._class == "Attribute": #  or tree._class == "Subscript" (handled above)
                if iscanon(tree.value):
                    (tl, tmpvar) = self.edispatch(tree.value)
                    tree.value = tmpvar

            elif tree._class == "UnaryOp":
                if iscanon(tree.operand):
                    (tl, tmpvar) = self.edispatch(tree.operand)
                    tree.operand = tmpvar

            elif tree._class == "BinOp" and tree.op == "**":
                (tl, tmpvar) = self.edispatch(tree.clone(), val=tree)
                tree = tmpvar

            elif tree._class == "BinOp":
                if iscanon(tree.left):
                    (tl, tmpvar) = self.edispatch(tree.left)
                    tree.left = tmpvar
                if iscanon(tree.right):
                    (tr, tmpvar) = self.edispatch(tree.right)
                    tree.right = tmpvar

            elif tree._class == "keyword" and False:
                if iscanon(tree.value):
                    (tl, tmpvar) = self.edispatch(tree.value)
                    tree.value = tmpvar

            elif tree._class == "List":
                tree.elts = [ self.edispatch(e)[1] if isop(e) else self.dispatch(e) for e in tree.elts ]

            res = tree
        else:
            res = tree
        return res

def canonicalize(tree, **kw):
    an = ASTCanonicalizer()
    return an(tree)


class ASTLocalAction:
    def __init__(self): pass
    def Before(self, t): pass
    def After(self, t): pass
    def Begin(self, t): pass
    def End(self, t): pass

    def __call__(self, tree):
        self.Begin(tree)
        result = self.dispatch(tree)
        er = self.End(tree)
        if er is not None:
            result = er
        return result

    def dispatch(self, tree):
        if type(tree) == type([]):
            res = list(map(self.dispatch, tree))
        elif isinstance(tree, ASTNode):

            res = tree
            br = self.Before(tree)
            if br is not None:
                b_res, b_ret = br
                res = b_res
                if b_ret:
                    return b_res

            for k in vars(res).keys():
                setattr(res, k, self.dispatch(getattr(res, k)))

            ar = self.After(res)
            if ar is not None:
                res = ar


        else:
            res = tree
        return res

class ASTReolvetmpvars(ASTLocalAction):

    def Begin(self, tree):
        self.seen = {}

    def Before(self, tree):
        if tree._class == "TmpVar":
            if tree.id in self.seen:
                short = self.seen[tree.id]
            else:
                short = f'{tree.kind}{len(self.seen):d}'
                self.seen[tree.id] = short
            tree = Name(f'{short}')
            return (tree, True)

def resolvetmpvars(tree, **kw):
    an = ASTReolvetmpvars()
    return an(tree)


class ASTPatchSuper(ASTLocalAction):

    def Begin(self, tree):
        self.func = None

    def Before(self, tree):
        if tree._class == "FunctionDef":
            self.func = tree
        elif tree._class == "Call" and getattr(tree.func, 'id', '') == "super":
            selfname = self.func.args.args[0].arg
            if len(tree.args) == 0:
                tree.args = [Attribute(Name(selfname), '__class__'), Name(selfname)]
            return (tree, True)


def mkOprevName(dict):
    rdict = {v: k.lower() for k, v in dict.items()}
    def inner(op):
        return rdict[op]
    return inner

def mkUnOprevName(dict):
    rdict = {k.lower(): v for k, v in dict.items()}
    def inner(op):
        return rdict[op]
    return inner

op_revname = mkOprevName(Unparser.binop | Unparser.boolops | Unparser.cmpops)
op_revname_unary = mkOprevName(Unparser.unop)

op_unrevname = mkUnOprevName(Unparser.unop | Unparser.binop | Unparser.boolops | Unparser.cmpops)

class ASTReplaceOps(ASTLocalAction):

    def __init__(self, **kw):
        self.replace = kw.get('replace', ['binop', 'unaryop', 'augassign'])
        #self.replace += ['attribute']

    def Begin(self, tree):
        self.seen = {}

    def Before(self, tree):
        pass

    def After(self, tree):
        transformed = True
        if tree._class == "BinOp" and 'binop' in self.replace:
            if tree.op == "*" and tree.left._class == "Constant":
                res = Call(f'binop_c_{op_revname(tree.op)}', [tree.left, tree.right])
            elif tree.op == "*" and tree.right._class == "Constant":
                res = Call(f'binop_d_{op_revname(tree.op)}', [tree.left, tree.right])
            else:
                res = Call(f'binop_{op_revname(tree.op)}', [tree.left, tree.right])
        elif tree._class == "UnaryOp" and 'unaryop' in self.replace:
            res = Call(f'unaryop_{op_revname_unary(tree.op)}', [tree.operand])
        elif tree._class == "CmpOp" and 'cmpop' in self.replace:
            res = Call(f'cmpop_{op_revname(tree.op)}', [tree.value])
        elif tree._class == "BoolOp" and 'boolop' in self.replace:
            res = Call(f'boolop_{op_revname(tree.op)}', tree.values)
        elif tree._class == "AugAssign" and 'augassign' in self.replace:
            res = Assign(tree.target, Call(f'binop_{op_revname(tree.op)}', [tree.target, tree.value]))
        elif tree._class == "Attribute" and 'attribute' in self.replace:
            res = Call(f'getattr', [tree.value, Constant(tree.attr)])
        else:
            transformed = False
            res = tree
        if transformed:
            res.transformed = True
        return res

class ASTReplaceOpsInvert(ASTLocalAction):

    def __init__(self, **kw):
        self.replace = kw.get('replace', ['binop', 'unaryop', 'augassign'])
        #self.replace += ['attribute']

    def Begin(self, tree):
        self.seen = {}

    def Before(self, tree):
        pass

    def After(self, tree):
        if getattr(tree, 'transformed', False):
            transformed = True
            try:
                if 'binop' in self.replace and tree._class == "Call" and getattr(tree.func, 'id', '').startswith('binop_'):
                    res = BinOp(op_unrevname(tree.func.id[6:]), tree.args[0], tree.args[1])
                elif 'unaryop' in self.replace and tree._class == "Call" and getattr(tree.func, 'id', '').startswith('unaryop_'):
                    res = UnaryOp(op_unrevname(tree.func.id[8:]), tree.args[0])
                elif 'cmpop' in self.replace and tree._class == "Call" and getattr(tree.func, 'id', '').startswith('cmpop_'):
                    res = CmpOp(op_unrevname(tree.func.id[6:]), tree.args[0], tree.args[1])
                elif 'boolop' in self.replace and tree._class == "Call" and getattr(tree.func, 'id', '').startswith('boolop_'):
                    res = BoolOp(op_unrevname(tree.func.id[7:]), tree.args[0], tree.args[1])
                elif 'augassign' in self.replace and tree._class == "Assign" \
                     and len(tree.targets) == 1 and tree.value._class == "Call" \
                     and getattr(tree.value.func, 'id', '').startswith('binop_'):
                    res = AugAssign(op_revname(tree.value.func.id), tree.targets[0], tree.value.args[1])
                elif 'attribute' in self.replace and tree._class == "Call" and tree.func.id == 'op_getattr':
                    res = Attribute(tree.args[0], tree.args[1].value)
                else:
                    transformed = False
                    res = tree
            except ValueError:
                transformed = False
                res = tree
            if transformed:
                res.untransformed = True
        else:
            res = tree
        return res


def normalize(tree, **kw):
    tree = ASTPatchSuper()(tree)
    if kw.get('replaceops', False):
        tree = ASTReplaceOps()(tree)
    #tree = resolvetmpvars(tree)
    #tree = astunparse.normalize(tree)
    return tree


def unnormalize(tree, **kw):
    if kw.get('replaceops', False):
        tree = ASTReplaceOpsInvert()(tree)
    return tree


class ASTVisitorLastFunction(ASTLocalAction):

    def Begin(self, tree):
        self.seen = []

    def Before(self, tree):
        if tree._class == "FunctionDef":
            self.seen.append((tree.name, tree))
            return (tree, True)

    def End(self, tree):
        lname, lfunc = self.seen[-1]
        return Module([lfunc]), lname


def filterLastFunction(intree):
    trans = ASTVisitorLastFunction()
    return trans(intree)


class ASTVisitorFilterFunctions(ASTLocalAction):
    def __init__(self, names):
        if isinstance(names, str):
            names = names.split('.')
        self.names = names

    def Begin(self, tree):
        self.seen = []
        self.index = 0
        self.pos = 0

    def Before(self, tree):
        # found, or to deep in tree, prune
        if self.pos > self.index or self.index >= len(self.names):
            return (tree, True)
        # print(f'XSearch {tree._class} for {self.names[self.index]}')
        if tree._class == "FunctionDef" or tree._class == "ClassDef":
            self.pos += 1
            if tree.name == self.names[self.index]:
                # print(f'XFound {tree._class[:-3]} with name {tree.name} at level {self.index}')
                if self.index == len(self.names) -1:
                    self.seen.append(tree.clone())
                    self.index += 1
                    return (tree, True)
                else:
                    self.index += 1

    def After(self, tree):
        if tree._class == "FunctionDef" or tree._class == "ClassDef":
            self.pos -= 1

    def End(self, tree):
        if len(self.seen) == 0:
            raise(NotFound('Class or function not found: ' + '.'.join(self.names)))
        return Module(self.seen)


def filterFunctions(intree, names):
    trans = ASTVisitorFilterFunctions(names)
    return trans(intree)


class ASTVisitorLastFunctionSig(ASTLocalAction):
    def Begin(self, tree):
        self.name = ''
        self.sig = []
        self.seen = []

    def Before(self, tree):
        if tree._class == "FunctionDef":
            self.sig = [t.arg for t in tree.args.args]
            self.name = tree.name
            return [tree, True]

    def End(self, tree):
        return (self.name, self.sig)


def infoSignature(intree):
    trans = ASTVisitorLastFunctionSig()
    return trans(intree)


class ASTVisitorImports(ASTLocalAction):
    def Begin(self, tree):
        self.imports = {}
        self.modules = []

    def Before(self, tree):
        if tree._class == "ImportFrom":
            mname = f'{tree.module}' if tree.module is not None else None
            for f in tree.names:
                if f.asname:
                    self.imports[f.asname] = {(mname, tree.level): f'{f.name}'}
                else:
                    self.imports[f.name] = {(mname, tree.level): f'{f.name}'}

        elif tree._class == "Import":
            for f in tree.names:
                if f.asname:
                    self.imports[f.asname] = f.name
                    self.modules.append(f.asname)
                else:
                    self.imports[f.name] = f.name
                    self.modules.append(f.name)

    def End(self, tree):
        return self.imports, self.modules


class ASTVisitorDict(ASTLocalAction):
    def Begin(self, tree):
        self.dict = {}
        self.path = []
        self.infunc = 0
        self.infuncs = []

    def Before(self, tree):
        if tree._class == "FunctionDef":
            self.infunc += 1
            if self.infunc > 1:
                self.path += ['<locals>.' + tree.name]
            else:
                self.path += [tree.name]
            self.dict['.'.join(self.path)] = tree
        elif tree._class == "ClassDef":
            self.infuncs += [self.infunc]
            self.infunc = 0
            self.path += [tree.name]
            self.dict['.'.join(self.path)] = tree

    def After(self, tree):
        if tree._class == "FunctionDef":
            self.infunc -= 1
            self.path = self.path[0:-1]
        elif tree._class == "ClassDef":
            self.path = self.path[0:-1]
            self.infunc = self.infuncs[-1]
            self.infuncs = self.infuncs[0:-1]


    def End(self, tree):
        return self.dict


class ASTVisitorLocals(ASTLocalAction):
    @classmethod
    def getRoot(self, t):
        if t._class == "Attribute" or t._class == "Subscript":
            return self.getRoot(t.value)
        return t

    @classmethod
    def getVars(self, t):
        res = []
        if isinstance(t, list) or isinstance(t, tuple):
            for e in t:
                res += ASTVisitorLocals.getVars(e)
        elif t._class == "Tuple":
            for e in t.elts:
                res += ASTVisitorLocals.getVars(e)
        elif t._class == "Name":
            res = [t.id]
        elif t._class == "arg":
            res = [t.arg]
        elif t._class == "Attribute" or t._class == "Subscript":
            res = [self.getRoot(t).id]
        return res

    def Begin(self, tree):
        self.locals = []
        self.localfuncs = []

    def Before(self, tree):
        if tree._class == "FunctionDef":
            self.localfuncs += [ tree.name ]
            self.locals += [ tree.name ]
            self.locals += [ n.arg for n in tree.args.args ]
            for deco in tree.decorator_list:
                self.locals += self.getVars(deco)
            if tree.args.kwarg:
                self.locals += [ tree.args.kwarg.arg ]
            if tree.args.vararg:
                self.locals += [ tree.args.vararg.arg ]

        elif tree._class == "Assign":
            for n in tree.targets:
                if n._class == "Name":
                    self.locals += [ n.id ]
                elif n._class == "Tuple":
                    self.locals += [ m.id for m in n.elts if m._class == "Name" ]

        elif tree._class == "For" or tree._class == "comprehension":
            self.locals += self.getVars(tree.target)

        elif tree._class == "With":
            self.locals += [ self.getRoot(s.optional_vars).id for s in tree.items if s.optional_vars is not None ]

    def End(self, tree):
        return self.locals, self.localfuncs


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
