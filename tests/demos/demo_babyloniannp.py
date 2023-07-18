import pyfad
import numpy as np

#vectorized gbabylonian
def gbabylonian(x, y=1):
    if np.all(np.abs(y**2 - x) < 1e-7):
        return y
    else:
        r = gbabylonian(x, (y + x / y) / 2)
        return r

def fbabylonian(x):
    r = gbabylonian(x)
    return r

# define input value
N = 3
x = np.arange(1, N + 1)

# compute function result
r = fbabylonian(x)
print(f'f{x}) = {r}')

#compute derivative
d_r, r = pyfad.DiffFor(fbabylonian, x, seed = [np.ones(N)])
# print(f'f{x}) = {r}')
print(f'f\'({x}) = {d_r},\n correct result is {0.5/r},\
 error {np.linalg.norm(d_r[0] - 0.5/r)}')
