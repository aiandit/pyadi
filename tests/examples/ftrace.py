
import astunparse
from . import fx

def floadast(x):
    print(astunparse.loadast(fx.f1))

