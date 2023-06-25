def f1(x,y,z):
    r = x*2*x*y*z
    return r

def f2(x,y,z):
    r = x*y*z*17
    return r

def f3(x,y,z):
    r = x*f1(y*z*17,y*z,z)
    return r
