import numpy as np

def gsqr(X):
    return np.matmul(X,X)

def fmatmul(x):
    M = np.diag(x)
    M2 = M @ M
    return np.sum(M2)

def fmatmul2(x):
    M = np.diag(x)
    M2 = np.matmul(M, M)
    return np.sum(M2)

def fmatmul3(x):
    M = np.diag(x)
    M2 = 2.1*M
    r = M @ M2 @ M
    return np.sum(np.diag(r))

def _fmatmul4(x):
    M = np.diag(x)
    M2 = 2.1*M
    print(M.shape, M2.shape)
    r = M @ np.invert(M2) @ M
    return np.sum(np.diag(r))

def fnorm(x):
    M = np.diag(x)
    M2 = 2.1*M
    r = np.linalg.norm(np.diag(M2))
    return r

def ftest(x):
    M = np.diag(x)
    M2 = np.matmul(M, 2*M)
    r = np.linalg.norm(np.diag(M2))
    return r

def fexp(x):
    v = x[0]
    z = np.exp(v)
    return z

def fexp2(x):
    v = x[0]
    w = x[1]
    z = np.exp(complex(v, w))
    return z

def fexp3(x):
    v = x[0]
    w = x[1]
    z = np.exp(v + 1j * w)
    return z
