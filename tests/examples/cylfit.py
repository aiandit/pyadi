import numpy as np
import pyfad
import os

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


def runobj():
    objComps, obj, handle = cylfit_obj()
    v0 = np.array([1.1, 0.01, 0.01])
    r = obj(v0)
    print (r)


def cyl2xyz(cx):
    N, _ = cx.shape
    cres = cx[:,0] * np.exp(1j * cx[:,1])
    return np.hstack([np.real(cres).reshape((N,1)), np.imag(cres).reshape((N,1)), cx[:,2].reshape((N,1))])


def mkCylData(N=10000, R0=1, theta=0, phi=0):
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


def runopt(fprime=None, gtol=1e-7):
    import scipy as sc

    R0 = 1.1
    theta0 = 0.1
    phi0 = np.pi/2

    # v0 = np.array([1, 0, 0])
    v0 = np.array([1.1, 0.01, 0.01])
    objComps, obj, handle = cylfit_obj()

    N = int(1e2)**2

    demopts = mkCylData(N, R0, theta0, phi0)

    handle()['points'] = demopts

    print('start fmin_cg')
    # does not work as soon as we give it a derivative??
    res = sc.optimize.fmin_cg(obj, v0, fprime=fprime, full_output=True, gtol=gtol, norm=2)
    print(res)

    sol, *rem = res

    sol[1] %= np.pi
    sol[2] %= np.pi

    rsol = obj(sol)

    errsol = relNormMax(sol, np.array([R0, theta0, phi0]))
    print(f'fmin_cg solution {sol}, expected {[R0, theta0, phi0]}')
    print(f'fmin_cg solution error {errsol}, final objective {rsol}')

def runopt_ad():

    objComps, obj, handle = cylfit_obj()

    def grad(*args, **kw):
        (dr, r) = pyfad.DiffFor(obj, *args)
        x = args[0]
        N = x.size
        g = np.zeros((N,))
        for i in range(N):
            g[i] = dr[i]
        print(f'gobj(x) = r={r}, g={g}')
        return g

    return runopt(grad, gtol=1e-10)


def runopt_fd():

    objComps, obj, handle = cylfit_obj()

    def grad(*args, **kw):
        (dr, r) = pyfad.DiffFD(obj, *args)
        x = args[0]
        N = x.size
        g = np.zeros((N,))
        for i in range(N):
            g[i] = dr[i]
        print(f'gobj(x) = r={r}, g={g}')
        return g

    return runopt(grad, gtol=1e-10)


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

    # NA: fsolve wants Nin == Nout?!
    print('start sc.optimize.fsolve')
    res = sc.optimize.fsolve(objComps, v0, fprime=fprime)
    print('sc.optimize.fsolve result', res, vs)

    sol = vs

    sol[1] *= -1
    sol[2] *= -1

    sol[1] %= np.pi
    sol[2] %= np.pi

    rsol = obj(sol)

    errsol = relNormMax(sol, np.array([R0, theta0, phi0]))
    print(f'fsolve solution {sol}, expected {[R0, theta0, phi0]}')
    print(f'fsolve solution error {errsol}, final objective {rsol}')


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


# run the UOpt uopt solver in CG mode (no Hessian)
# Cannot run alone, derivatives must be provided
def _runuopt(fprime=None):
    import uopt.uopt

    objComps, obj1, handle = cylfit_obj()

    def fobj(x, y, udata):
        r = obj1(x)
        y[:] = r

    def gobj(x, y, g, udata):
        fprime(obj1, x, y, g, udata)

    R0 = 1.1
    theta0 = 0.1
    phi0 = np.pi/2

    # v0 = np.array([1, 0, 0])
    v0 = np.array([1.1, 0.01, 0.01])

    N = int(1e2)**2

    demopts = mkCylData(N, R0, theta0, phi0)

    handle()['points'] = demopts

    r0 = obj1(v0)

    vs = v0.copy()
    rs = r0.copy()

    status = uopt.statusHist()

    print('start uopt')
    res = uopt.uopt(v0, vs, rs, fobj, gobj if fprime is not None else None,
                    s = status, opts=dict(tolObjAbs=1e-6, cgmode=2))
    print(res)

    sol = vs

    sol[1] *= -1
    sol[2] *= -1

    sol[1] %= np.pi
    sol[2] %= np.pi

    rsol = obj1(sol)

    errsol = relNormMax(sol, np.array([R0, theta0, phi0]))
    print(f'uopt CG solution {sol}, expected {[R0, theta0, phi0]}')
    print(f'uopt CG solution error {errsol}, final objective {rsol}')

    if not os.path.exists('plot'):
        os.mkdir('plot')

    uopt.mkPlot(status, 'J', 'it', outdir='plot')
    uopt.mkPlot(status, 'Er', 'it', outdir='plot')
    uopt.mkPlot(status, 'J', 'tj', outdir='plot')
    uopt.mkPlot(status, 'Er', 'tj', outdir='plot')


def runuopt_ad():

    def gobj(fobj, x, y, g, udata):
        (dr, r) = pyfad.DiffFor(fobj, x, verbose=0)
        y[:] = r
        for i in range(x.size):
            g[i] = dr[i]
        print(f'gobj(x) = r={r}, g={g}')
        return g, r

    _runuopt(gobj)


def runuopt_fd():

    def gobj(fobj, x, y, g, udata):
        (dr, r) = pyfad.DiffFD(fobj, x)
        y[:] = r
        for i in range(x.size):
            g[i] = dr[i]
        print(f'gobj(x) = r={r}, g={g}')
        return g, r

    _runuopt(gobj)


# run UOpt usolve. Cannot run alone, derivatives must be provided
def _runusolve(fprime=None, fvprime=None):
    import uopt.uopt

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

    N = int(1e2)**2

    demopts = mkCylData(N, R0, theta0, phi0)

    #demopts += (np.random.rand(N, 3) - 0.5) * 1e-6

    handle()['points'] = demopts

    r0 = objComps(v0)

    vs = v0.copy()
    rs = r0.copy()

    status = uopt.statusHist()

    print('start usolve')
    res = uopt.usolve(v0, vs, rs,
                      fobj,
                      gobj if fprime is not None else None,
                      gvobj if fvprime is not None else None,
                      s = status)
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

    if not os.path.exists('plot'):
        os.mkdir('plot')

    uopt.mkPlot(status, 'J', 'it', outdir='plot')
    uopt.mkPlot(status, 'Er', 'it', outdir='plot')
    uopt.mkPlot(status, 'J', 'tj', outdir='plot')
    uopt.mkPlot(status, 'Er', 'tj', outdir='plot')

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
            # because the function is very nonlinear, the solver
            # complains about a wrong derivative when we use DiffFD(f, x, seed=[dx])
            # so we have to compute the full (N, 3)-Jacobian and multiply by hand!
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

    def gobj(fobj, x, y, g, udata):
        (dr, r) = pyfad.DiffFor(fobj, x)
        y[:] = r
        for i in range(x.size):
            g[:,i] = dr[i]
        print(f'gobj(x) = {np.linalg.norm(y)}')

    def gvobj(fobj, x, y, dx, g, udata):
        # with AD the directional derivative is of course correct!
        (dr, r) = pyfad.DiffFor(fobj, x, seed=[dx])
        y[:] = r
        N, Ndd = dx.shape
        for i in range(Ndd):
            g[:,i] = dr[i]
        print(f'gvobj(x) = {np.linalg.norm(y)}')

    _runusolve(gobj, gvobj)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print(f'run as {sys.argv[0]} <mode>')
        modes = [f[3:] for f in dir(sys.modules[__name__]) if f.startswith('run')]
        mode = modes[-1]
        print(f'available modes: {modes}, run default: {mode}')
    else:
        mode = sys.argv[1]
    runf = getattr(sys.modules[__name__], 'run' + mode)
    runf()
    # runusolve()
    # runuopt()
    #runopt_ad()
    #runopt()
