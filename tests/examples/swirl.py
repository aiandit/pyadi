import numpy as np

def initialize_starting_point(nint):
    # INITIALIZE_STARTING_POINT    Computes the standard starting point
    #    for the swirling flow between two disks.
    #
    #    X = INITIALIZE_STARTING_POINT(NINT) computes the starting
    #    point X whose length is 14*NINT where NINT is the number
    #    of intervals used in the collocation method.

    # STCS 2007 at RWTH Aachen University
    # 04/20/07 by A. Vehreschild and H. Martin Buecker based on
    #  the implementation of the MINPACK-2 routine DSFDFJ by
    #  Brett M. Averick, 1993.
    # 2023 translated to Python by Johannes Willkomm

    cpts = 4;
    fdeg = 4;
    gdeg = 2;
    npi = 2*cpts+gdeg+fdeg;
    omega1 = -1;
    omega2 = 1;

    n = 14*nint;
    h = 1 / nint;
    x = np.zeros((n,));

    adr = np.arange(0, n, npi) + fdeg + cpts;
    x[adr] = omega1 + (omega2-omega1) * np.arange(nint) * h;
    x[adr+1] = omega2 - omega1;

    return x


def swirl(x, epsilon):
    # SWIRL    Defines a system of nonlinear equations f(x) = 0
    #    where f: R^n  ->  R^n represents the residuals. The
    #    nonlinear system describes the swirling flow between
    #    two disks. More precisely, the steady flow of viscous,
    #    incompressible, axisymmetric fluid between two rotating
    #    infinite coaxial disks leads to a system of two boundary
    #    value problems that are solved by a collocation method.
    #
    #    FVEC = SWIRL(X,EPSILON) evaluates the function at a
    #    vector X and returns the function value in FVEC. The
    #    length of both vectors is a multiple of 14. The parameter
    #    EPSILON describes the viscosity of the fluid.
    #
    #    See Problem 2.2 of the following report for more details:
    #       B.M. Averick, R.G. Carter, J.J. More and G.-L. Xue.
    #       The MINPACK-2 Test Problem Collection, Preprint MCS-P-153-0692,
    #       Mathematics and Computer Science Division, Argonne National
    #       Laboratory, 1992.
    #


    # STCS 2007 at RWTH Aachen University
    # 04/20/07 by B. Kuhlmann and H. Martin Buecker based on
    #  the implementation of the MINPACK-2 routine DSFDFJ by
    #  Brett M. Averick, 1993.
    # 2023 translated to Python by Johannes Willkomm

    n = len(x);
    bc = 3;
    cpts = 4;
    fdeg = 4;
    gdeg = 2;
    mdeg = 4;
    dim = mdeg + cpts - 1;
    npi = 2*cpts+gdeg+fdeg;
    omega1 = -1;
    omega2 = 1;
    rhnfhk = np.zeros((cpts, dim+1, dim+1, mdeg+1))
    rho = np.zeros((cpts,))
    wg = np.zeros((gdeg+1,))
    wf= np.zeros((fdeg+1,))
    fvec = np.zeros((n,))

    rho[0]=0.694318413734436035e-1;
    rho[1]=0.330009490251541138e0;
    rho[2]=0.669990539550781250e0;
    rho[3]=0.930568158626556396e0;

    # Check input arguments for errors
    if n % 14 != 0:
        error('ERROR: n not a multiple of 14');


    # Initialization
    nint = n // 14;
    h = 1 / nint;

    #Store all possible combinations of rho, h, and n factorial.
    hm = 1;
    for m in range(mdeg+1):
        for i in range(cpts):
            rhoijh = hm;
            for j in range(dim+1):
                nf = 1;
                for k in range(dim+1):
                    rhnfhk[i,j,k,m] = rhoijh / nf
                    nf = nf*(k+1);

                rhoijh = rhoijh*rho[i];


        hm = hm*h;

    #Set up the boundary equations at t = 0
    # f(0) = 0, f'(0) = 0, g(0) = omega1
    fvec[0] = x[0];
    fvec[1] = x[1];
    fvec[2] = x[cpts + fdeg] - omega1;

    #Set up the collocation equations
    for i in range(nint):
        var1 = i*npi;
        eqn1 = var1 + bc;
        var2 = var1 + cpts +fdeg;
        eqn2 = eqn1 + cpts;
        for k in range(cpts):
            for m in range(fdeg+1):
                wf[m] = 0;
                for j in range(m, fdeg):
                    wf[m] = wf[m] + rhnfhk[k,j-m,j-m,j-m]*x[var1+j];

                for j in range(cpts):
                    wf[m] = wf[m] + x[var1+fdeg+j]*rhnfhk[k,fdeg+j-m,fdeg+j-m,fdeg-m];


            for m in range(gdeg+1):
                wg[m] = 0;
                for j in range(m, gdeg):
                    wg[m] = wg[m] + rhnfhk[k, j-m, j-m, j-m]*x[var2+j];

                for j in range(cpts):
                    wg[m] = wg[m] + x[var2+gdeg+j]*rhnfhk[k,gdeg+j-m,gdeg+j-m,gdeg-m+1];


            fvec[eqn1+k] = epsilon*wf[4] + wf[3]*wf[0] + wg[1]*wg[0];
            fvec[eqn2+k] = epsilon*wg[2] + wf[0]*wg[1] - wf[1]*wg[0];



    #Set up the continuity equations
    for i in range(nint-1):
        var1 = i*npi;
        eqn1 = var1 + bc + 2*cpts;
        var2 = var1 + fdeg + cpts;
        eqn2 = eqn1 + fdeg;
        for m in range(fdeg):
            wf[m] = 0;
            for j in range(m, gdeg):
                wf[m] = wf[m] + rhnfhk[0,0,j-m,j-m]*x[var1+j];

            for j in range(cpts):
                wf[m] = wf[m] + rhnfhk[0,0,fdeg+j-m,fdeg-m]*x[var1+fdeg+j];


        for m in range(gdeg):
            wg[m] = 0;
            for j in range(m, gdeg):
                wg[m] = wg[m] + rhnfhk[0,0,j-m,j-m]*x[var2+j];

            for j in range(cpts):
                wg[m] = wg[m] + rhnfhk[0,0,gdeg+j-m,gdeg-m]*x[var2+gdeg+j];


        for m in range(fdeg):
            fvec[eqn1+m] = x[var1+npi+m] - wf[m];

        for m in range(gdeg):
            fvec[eqn2+m] = x[var2+npi+m] - wg[m];



    #     Prepare for setting up the boundary conditions at t = 1.
    var1 = n - npi
    for m in range(fdeg + 1):
        wf[m] = 0;
        for j in range(fdeg - m):
            wf[m] = wf[m] + rhnfhk[0,0,j,j+1]*x[var1+j+m];

        for j in range(cpts):
            wf[m] = wf[m] + rhnfhk[0,0,fdeg+j-m,fdeg-m]*x[var1+fdeg+j];


    var2 = var1 + fdeg + cpts
    for m in range(gdeg + 1):
        wg[m] = 0;
        for j in range(gdeg - m):
            wg[m] = wg[m] + rhnfhk[0,0,j,j]*x[var2+j+m];

        for j in range(cpts):
            wg[m] = wg[m] + rhnfhk[0,0,gdeg+j-m,gdeg-m+2]*x[var2+gdeg+j];



    #     Set up the boundary equations at t = 1.
    #     f(1) = 0, f'(1) = 0, g(1) = omega2.
    fvec[n-3] = wf[0];
    fvec[n-2] = wf[1];
    fvec[n-1] = wg[0] - omega2;

    return fvec


def run():
    init = initialize_starting_point(4)
    res = swirl(init, 1e-2)

    with open('init.txt', 'w') as f:
        np.savetxt(f, init)
    with open('swirl.txt', 'w') as f:
        np.savetxt(f, res)

if __name__ == "__main__":
    run()
