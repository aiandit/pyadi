import numpy as np
import pyfad

# https://stackoverflow.com/questions/48265646/rotation-of-a-vector-python
# vectorized


def x_rotation(vector,theta):
    """Rotates (N,3) array around x-axis, take -theta because we multiply from the right"""
    R = np.array([[1,0,0],[0,np.cos(-theta),-np.sin(-theta)],[0, np.sin(-theta), np.cos(-theta)]])
    return vector @ R

def y_rotation(vector,theta):
    """Rotates (N,3) array around y-axis, take -theta because we multiply from the right"""
    R = np.array([[np.cos(-theta),0,np.sin(-theta)],[0,1,0],[-np.sin(-theta), 0, np.cos(-theta)]])
    return vector @ R

def z_rotation(vector,theta):
    """Rotates (N,3) array around z-axis, take -theta because we multiply from the right"""
    R = np.array([[np.cos(-theta), -np.sin(-theta),0],[np.sin(-theta), np.cos(-theta),0],[0,0,1]])
    return vector @ R


def cylfit_obj():
    data = {}

    data['points'] = np.random.rand(100, 3) - 0.5

    def objComps(x):

        R = x[0]
        theta = x[1]
        phi = x[2]

        zaxis = np.zeros(3,)
        zaxis[2] = 1
        caxis = z_rotation(x_rotation(zaxis, theta), phi)
        caxis = caxis.reshape((1,3))

        pts = data['points']
        N, _ = pts.shape
        assert _ == 3

        ptoffs = np.sum(caxis * pts, axis=1)

        ptoffs = ptoffs.reshape((N, 1))
        ptinters = ptoffs * caxis

        pdvs = pts - ptinters
        dists = np.sqrt(np.sum(pdvs**2, axis=1))

        dists -= R

        assert pdvs.shape == (N,3)
        assert dists.shape == (N,)

        return dists

    def objv(x):

        dists = objComps(x)

        r = np.linalg.norm(dists)

        # print(f'res = {r}')
        return r

    def handle():
        return data

    return objComps, objv, handle


def run():
    objComps, obj, handle = cylfit_obj()
    v0 = np.array([1.1, 0.01, 0.01])
    r = obj(v0)
    print (r)


def cyl2xyz(cx):
    N, _ = cx.shape
    cres = cx[:,0] * np.exp(1j * cx[:,1])
    return np.hstack([np.real(cres).reshape((N,1)), np.imag(cres).reshape((N,1)), cx[:,2].reshape((N,1))])

def mkCylData(N=10000, R0=1, theta=0, phi=0):
    zaxis = np.zeros((1, 3))
    zaxis[0,2] = 1
    Ns = int(np.sqrt(N))
    N = Ns ** 2
    phis = np.linspace(0, 2*np.pi, Ns)
    heights = np.linspace(-1, 1, Ns)
    PP, HH = np.meshgrid(phis, heights)
    cpts = np.hstack([ R0 * np.ones((N,1)), PP.reshape((N,1)), HH.reshape((N,1)) ])
    pts = cyl2xyz(cpts)
    pts = z_rotation(x_rotation(pts, theta), phi)
    #print('pts generated')
    return pts


def relNormMax(r1, r2):
    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)
    nd = np.linalg.norm(r1 - r2)
    divi = max(n1, n2)
    if divi == 0: divi = 1
    return nd / divi


def runopt(fprime=None):
    import scipy as sc

    R0 = 1.1
    theta0 = 0.1
    phi0 = np.pi/2

    # v0 = np.array([1, 0, 0])
    v0 = np.array([1.1, 0.01, 0.01])
    objComps, obj, handle = cylfit_obj()

    N = int(1e3)**2

    demopts = mkCylData(N, R0, theta0, phi0)

    handle()['points'] = demopts

    print('start cg')
    res = sc.optimize.fmin_cg(obj, v0, fprime=fprime, full_output=True)
    print(res)

    sol, *rem = res

    rsol = obj(sol)

    errsol = relNormMax(sol, np.array([R0, theta0, phi0]))
    print(f'CG solution {sol}, expected {[R0, theta0, phi0]}')
    print(f'CG solution error {errsol}, final objective {rsol}')

def runopt_ad():

    objComps, obj, handle = cylfit_obj()

    def grad(*args, **kw):
        (dr, r) = pyfad.DiffFor(obj, *args)
        x = args[0]
        N = x.size
        g = np.zeros((N,))
        for i in range(N):
            g[i] = dr[i]
        return g

    return runopt(grad)




def runfsolve(fprime=None):
    import scipy as sc

    objComps, obj, handle = cylfit_obj()

    R0 = 1.1
    theta0 = 0.1
    phi0 = np.pi/2

    # v0 = np.array([1, 0, 0])
    v0 = np.array([1, 0.01, 0.01])

    N = int(1e3)**2

    demopts = mkCylData(N, R0, theta0, phi0)

    handle()['points'] = demopts

    r0 = objComps(v0)

    vs = v0.copy()
    rs = r0.copy()

    print('start usolve')
    res = sc.optimize.fsolve(objComps, v0, fprime=fprime)
    print('usolve result', res, vs)

    sol = vs

    sol[1] *= -1
    sol[2] *= -1

    sol[1] %= np.pi
    sol[2] %= np.pi

    rsol = obj(sol)

    errsol = relNormMax(sol, np.array([R0, theta0, phi0]))
    print(f'usolve solution {sol}, expected {[R0, theta0, phi0]}')
    print(f'usolve solution error {errsol}, final objective {rsol}')


def runfsolve_fd(fprime=None):
    objComps, obj, handle = cylfit_obj()

    def gobj(x, udata):
        (dr, r) = pyfad.DiffFDNP(objComps, x)
        g = np.zeros(r.size, x.size)
        for i in range(x.size):
            g[:,i] = dr[i]
        print(f'gobj(x) = {np.linalg.norm(r)}')
        return g

    runfsolve(gobj)


def runfsolve_ad(fprime=None):
    objComps, obj, handle = cylfit_obj()

    def gobj(x, udata):
        (dr, r) = pyfad.DiffFor(objComps, x)
        g = np.zeros(r.size, x.size)
        for i in range(x.size):
            g[:,i] = dr[i]
        print(f'gobj(x) = {np.linalg.norm(r)}')
        return g

    runfsolve(gobj)


def runuopt(fprime=None):
    import uopt.uopt
    import scipy as sc

    objComps, obj1, handle = cylfit_obj()

    def fobj_uopt(x, y, udata):
        r = obj1(x)
        y[:] = r

    def gobj_uopt(x, y, g, udata):
        (dr, r) = pyfad.DiffFor(obj1, x, verbose=2)
        y[:] = r
        for i in range(x.size):
            g[i] = dr[i]
        print(f'gobj(x) = r={r}, g={g}')
        return g, r

    R0 = 1.1
    theta0 = 0.1
    phi0 = np.pi/2

    # v0 = np.array([1, 0, 0])
    v0 = np.array([1.1, 0.01, 0.01])

    N = int(1e3)**2

    demopts = mkCylData(N, R0, theta0, phi0)

    handle()['points'] = demopts

    r0 = obj1(v0)

    vs = v0.copy()
    rs = r0.copy()

    print('start uopt')
    res = uopt.uopt.uopt(v0, vs, rs, fobj_uopt, gobj_uopt)
    print(res)

    sol = vs
    rsol = obj(sol)

    errsol = relNormMax(sol, np.array([R0, theta0, phi0]))
    print(f'CG solution {sol}, expected {[R0, theta0, phi0]}')
    print(f'CG solution error {errsol}, final objective {rsol}')


def _runusolve(fprime=None, fvprime=None):
    import uopt.uopt
    import scipy as sc

    objComps, obj, handle = cylfit_obj()

    def fobj(x, y, udata):
        y[:] = objComps(x)
        print(f'obj(x) = {np.linalg.norm(y)}')

    def gobj(x, y, g, udata):
        fprime(objComps, x, y, g, udata)

    def gvobj(x, y, dx, g, udata):
        fvprime(objComps, x, y, dx, g, udata)

    R0 = 1.1
    theta0 = 0.1
    phi0 = np.pi/2

    # v0 = np.array([1, 0, 0])
    v0 = np.array([1, 0.01, 0.01])

    N = int(1e3)**2

    demopts = mkCylData(N, R0, theta0, phi0)

    #demopts += (np.random.rand(N, 3) - 0.5) * 1e-6

    handle()['points'] = demopts

    r0 = objComps(v0)

    vs = v0.copy()
    rs = r0.copy()

    print('start usolve')
    res = uopt.uopt.usolve(v0, vs, rs,
                           fobj,
                           gobj if fprime is not None else None,
                           gvobj if fvprime is not None else None)
    print('usolve result', res, vs)

    sol = vs

    sol[1] *= -1
    sol[2] *= -1

    sol[1] %= np.pi
    sol[2] %= np.pi

    rsol = obj(sol)

    errsol = relNormMax(sol, np.array([R0, theta0, phi0]))
    print(f'usolve solution {sol}, expected {[R0, theta0, phi0]}')
    print(f'usolve solution error {errsol}, final objective {rsol}')


def runusolve_fd():

    def gobj(fobj, x, y, g, udata):
        (dr, r) = pyfad.DiffFD(fobj, x)
        y[:] = r
        for i in range(x.size):
            g[:,i] = dr[i]
        print(f'gobj(x) = {np.linalg.norm(y)}')

    def gvobj(fobj, x, y, dx, g, udata):
        #print(f'dx={dx.shape}, g={g.shape}')
        if True:
            (drF, rF) = pyfad.DiffFD(fobj, x)
            Jac = np.zeros((rF.size, x.size))
            for i in range(x.size):
                Jac[:,i] = drF[i]
            Jv = Jac @ dx
            if False:
                print('check Jac', Jac)
                print('check dx', dx)
                print('check', Jv)
                (drnp, rnp) = pyfad.DiffFDNP(fobj, x, seed=[dx ])
                print('check fd2', drnp)
                (drnp2, rnp2) = pyfad.DiffFDNP(lambda t: fobj(x + t*dx.flat), np.array([0.0]))
                print('check fd2', drnp2)
                (dr, r) = pyfad.DiffFD(fobj, x, seed=[dx ])
        y[:] = rF
        g[:] = Jv
        print(f'gvobj(x) = {np.linalg.norm(y)}')

    _runusolve(gobj, gvobj)


def runusolve_ad():
    import uopt.uopt
    import scipy as sc

    objComps, obj, handle = cylfit_obj()

    def gobj(fobj, x, y, g, udata):
        (dr, r) = pyfad.DiffFor(fobj, x)
        y[:] = r
        for i in range(x.size):
            g[:,i] = dr[i]
        print(f'gobj(x) = {np.linalg.norm(y)}')

    def gvobj(fobj, x, y, dx, g, udata):
        (dr, r) = pyfad.DiffFor(fobj, x, seed=[dx])
        y[:] = r
        N, Ndd = dx.shape
        for i in range(Ndd):
            g[:,i] = dr[i]
        print(f'gvobj(x) = {np.linalg.norm(y)}')

    _runusolve(gobj, gvobj)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'usolve'
    runf = getattr(sys.modules[__name__], 'run' + mode)
    runf()
    # runusolve()
    # runuopt()
    #runopt_ad()
    #runopt()
