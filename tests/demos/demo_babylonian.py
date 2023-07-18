import pyfad

def gbabylonian(x, y=1):
    if abs(y**2 - x) < 1e-7:
        return y
    else:
        r = gbabylonian(x, (y + x / y) / 2)
        return r

def fbabylonian(x):
    r = gbabylonian(abs(x))
    return r

# define input value
x = 16

# compute function result
r = fbabylonian(x)
print(f'f({x}) = {r}')

#compute derivative
d_r, r = pyfad.DiffFor(fbabylonian, x)
#print(f'f({x}) = {r}')
print(f'f\'({x}) = {d_r}, correct result is {0.5/r}, error {d_r[0] - 0.5/r}')
