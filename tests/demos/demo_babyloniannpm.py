import pyfad
import numpy as np

#vectorized gbabylonian
def gbabylonian(x, y):
    if np.all(np.abs(y @ y - x) < 1e-7):
        return y
    else:
        r = gbabylonian(x, (y + x @ np.linalg.inv(y)) / 2)
        return r

def fbabylonian(x):
    r = gbabylonian(x, np.eye(*x.shape))
    return r

# define input value
N = 2
x = 2 * np.random.rand(N, N) - 0.5

# compute function result
while True:
    try:
        # at least works sometimes
        r = fbabylonian(x)
        break
    except:
        x = 2 * np.random.rand(N, N) - 0.5
        pass

print(f'f({x}) = {r}, check: {r@r}, error {np.linalg.norm(x - r@r)}')

#compute derivative
d_r, r = pyfad.DiffFor(fbabylonian, x)
# print(f'f({x}) = {r}')
print(f'f\'({x}) = {d_r}')
