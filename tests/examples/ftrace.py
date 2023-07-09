import argparse
import astunparse
from . import fx

def floadast(x):
    return astunparse.loadast(fx.f1)

def gunparse(x):
    return astunparse.unparse(x)

def pargs(x):

    parser = argparse.ArgumentParser(
                    prog='prog',
                    description='Descr')
    parser.add_argument('filename', nargs='?')
    parser.add_argument('-F', '--function', type=str, dest='activef')
    parser.add_argument('-p', '--prefix', type=str, nargs='*')
    parser.add_argument('-I', '--independents', type=str, dest='active', nargs='*')
    parser.add_argument('-i', '--indent', type=int, default=0, const=1, nargs='?')
    parser.add_argument('-o', '--output', type=str,
                        help='Write output to file')
    parser.add_argument('-g', '--debug', type=str, nargs='?', const='x',
                        help='Keep line number information')
    parser.add_argument('-v', '--verbose',
                        action='store_true')

    args = parser.parse_args(['test.bin', '-F', 'func', '-o', 'out.txt'])
