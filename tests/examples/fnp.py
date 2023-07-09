import numpy as np

def fsqr(X):
    return np.matmul(X,X)

def fmatmul(x):
    l = [x, x*x, x*x*x]
    M = np.diag(l)
    M2 = M @ M
    return np.sum(M2)

