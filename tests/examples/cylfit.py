import numpy as np

# https://stackoverflow.com/questions/48265646/rotation-of-a-vector-python
# vectorized

def x_rotation(vector,theta):
    """Rotates 3-D vector around x-axis"""
    R = np.exp(1j * theta)
    N, _ = vector.shape
    assert _ == 3
    tmp = vector[:,1] + 1j * vector[:,1]]
    tmp[
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.sum(R * vector, axis=1)

def y_rotation(vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.sum(R * vector, axis=1)

def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.sum(R * vector, axis=1)


def cylfit_obj():
    data = {}

    data['points'] = np.random.rand(100, 3) - 0.5

    def obj(x):

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

        r = np.linalg.norm(dists)

        print(f'res = {r}')
        return r

    def handle():
        return data

    return obj, handle


def run():
    obj, handle = cylfit_obj()
    v0 = np.array([1.1, 0.01, 0.01])
    r = obj(v0)
    print (r)


def cyl2xyz(cx):
    N, _ = cx.shape
    cres = cx[:,0] * np.exp(1j * cx[:,1])
    return np.hstack([np.real(cres).reshape((N,1)), np.imag(cres).reshape((N,1)), cx[:,2].reshape((N,1))])

def mkCylData(N=10000, theta=0, phi=0):
    zaxis = np.zeros((1, 3))
    zaxis[0,2] = 1
    Ns = int(np.sqrt(N))
    phis = np.linspace(0, 2*np.pi, Ns)
    heights = np.linspace(-1, 1, Ns)
    PP, HH = np.meshgrid(phis, heights)
    cpts = np.hstack([ np.ones((N,1)), PP.reshape((N,1)), HH.reshape((N,1)) ])
    pts = cyl2xyz(cpts)
    pts = z_rotation(x_rotation(pts, theta), phi)
    return pts

def runopt():
    import scipy as sc

    v0 = np.array([1, 0, 0])
    #v0 = np.array([1.1, 0.01, 0.01])
    obj, handle = cylfit_obj()

    N = int(1e3)**2

    demopts = mkCylData(N)

    handle()['points'] = demopts

    res = sc.optimize.fmin_cg(obj, v0, full_output=True)
    print(res)

if __name__ == "__main__":
    runopt()
