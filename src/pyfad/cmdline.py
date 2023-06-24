import sys, os, argparse

from . import *
from astunparse import unparse, unparse2j, unparse2x, loadastpy

def pycanon():
    run(lambda x, y, **kw: unparse(normalize(canonicalize(loadastpy(x, **kw)), **kw)),
        prog="pycanon", description="Canonicalize Python source code")

def pydiff():
    run(lambda x, y, **kw: unparse(differentiate(loadastpy(x, **kw), **kw)),
        prog="pydiff", description="Differentiate Python source code")

def run(parsefun, prog, description='What the program does'):
    parser = argparse.ArgumentParser(
                    prog=prog,
                    description=description)
    parser.add_argument('filename', nargs='?')
    parser.add_argument('-F', '--functions', type=str, dest='activef', nargs='*')
    parser.add_argument('-I', '--independents', type=str, dest='active', nargs='*')
    parser.add_argument('-i', '--indent', type=int, default=0, const=1, nargs='?')
    parser.add_argument('-o', '--output', type=str,
                        help='Write output to file')
    parser.add_argument('-g', '--debug', type=str, nargs='?', const='x',
                        help='Keep line number information')
    parser.add_argument('-v', '--verbose',
                        action='store_true')

    args = parser.parse_args()

    input = ''
    if not args.filename:
        input = sys.stdin.read()
        fname = 'stdin'
    else:
        fname = args.filename
        input = open(args.filename).read()
    out = sys.stdout
    if args.output:
        out = open(args.output, 'w')
    debug = True if args.debug else False
    indent = args.indent

    if isinstance(parsefun, list):
        res = ''
        for i, pfun in enumerate(parsefun):
            res = pfun(input, fname, indent=indent, debug=debug)
            if debug:
                if isinstance(res, str):
                    with open(fname + '.' + f'{i}', 'w') as f:
                        f.write(res)
            input = res
        print(res, file=out)
    else:
        print(parsefun(input, fname, **vars(args)), file=out)
