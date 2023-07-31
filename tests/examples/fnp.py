
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

def fmatmul4(x):
    M = np.diag(x)
    M2 = 2.1*M
    #print(M.shape, M2.shape)
    r = M @ np.linalg.inv(M2) @ M
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

def fmnorm(x):
    M = np.diag(x)
    z = np.linalg.norm(M)
    return z

def fmnorm2(x):
    M = np.diag(x)
    z = np.linalg.norm(M, 2)
    return z

def fmnorm3(x):
    M = np.diag(x)
    I = np.eye(*M.shape)
    z = np.linalg.norm(M @ (M + I), 2)
    return z

def fmnorm4(x):
    M = np.diag(x)
    I = np.ones(M.size)
    I = I.reshape(*M.shape)
    z = np.linalg.norm(M @ (M + I))
    return z

def fmnorm5(x):
    M = np.diag(x)
    I = np.ones(M.size)
    I = I.reshape(*M.shape)
    z = np.linalg.norm(M @ (M + I), 2)
    return z

def fmnorm6(x):
    M = np.diag(x)
    I = np.ones(M.size)*1e-3
    z = np.linalg.norm(M @ (M + I.reshape(*M.shape)), 2)
    return z

g_X = np.ones(9)*1e-3

def _fmnorm7(x):
    M = np.diag(x)
    I = np.ones(M.size)*1e-3
    z = np.linalg.norm(M @ (M + g_X.reshape(*M.shape)), 2)
    return z
