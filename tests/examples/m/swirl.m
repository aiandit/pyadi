function fvec = swirl(x,epsilon)
% SWIRL    Defines a system of nonlinear equations f(x) = 0
%    where f: R^n  ->  R^n represents the residuals. The
%    nonlinear system describes the swirling flow between
%    two disks. More precisely, the steady flow of viscous,
%    incompressible, axisymmetric fluid between two rotating
%    infinite coaxial disks leads to a system of two boundary
%    value problems that are solved by a collocation method.
%
%    FVEC = SWIRL(X,EPSILON) evaluates the function at a
%    vector X and returns the function value in FVEC. The
%    length of both vectors is a multiple of 14. The parameter
%    EPSILON describes the viscosity of the fluid.
%
%    See Problem 2.2 of the following report for more details:
%       B.M. Averick, R.G. Carter, J.J. More and G.-L. Xue.
%       The MINPACK-2 Test Problem Collection, Preprint MCS-P-153-0692,
%       Mathematics and Computer Science Division, Argonne National
%       Laboratory, 1992.
%


% STCS 2007 at RWTH Aachen University
% 04/20/07 by B. Kuhlmann and H. Martin Buecker based on
%  the implementation of the MINPACK-2 routine DSFDFJ by
%  Brett M. Averick, 1993.

n = length(x);
bc = 3;
cpts = 4;
fdeg = 4;
gdeg = 2;
mdeg = 4;
dim = mdeg + cpts - 1;
npi = 2*cpts+gdeg+fdeg;
omega1 = -1;
omega2 = 1;
rhnfhk = zeros(cpts, dim+1, dim+1, mdeg+1);
rho = zeros(cpts,1);
wg = zeros(gdeg+1,1);
wf= zeros(fdeg+1,1);
fvec = zeros(n,1);

rho(1)=0.694318413734436035d-1;
rho(2)=0.330009490251541138d0;
rho(3)=0.669990539550781250d0;
rho(4)=0.930568158626556396d0;

% Check input arguments for errors
if mod(n,14) ~= 0;
    error('ERROR: n not a multiple of 14');
end

% Initialization
nint = n / 14;
h = 1 / nint;

%Store all possible combinations of rho, h, and n factorial.
hm = 1;
for m = 0: mdeg
    for i = 1 : cpts
        rhoijh = hm;
        for j = 0 : dim;
            nf = 1;
            for k = 0 : dim;
                rhnfhk(i,j+1,k+1,m+1) = rhoijh / nf;
                nf = nf*(k+1);
            end;
            rhoijh = rhoijh*rho(i);
        end;
    end;
    hm = hm*h;
end;

srhnfhk = permute(rhnfhk, fliplr(1:ndims(rhnfhk)))(:);
save -ascii rhnfhkm.txt srhnfhk

%Set up the boundary equations at t = 0
% f(0) = 0, f'(0) = 0, g(0) = omega1
fvec(1) = x(1);
fvec(2) = x(2);
fvec(3) = x(cpts + fdeg + 1) - omega1;

%Set up the collocation equations
for i = 1 : nint;
    var1 = (i-1)*npi;
    eqn1 = var1 + bc;
    var2 = var1 + cpts +fdeg;
    eqn2 = eqn1 + cpts;
    for k = 1 : cpts;
        for m = 1 : fdeg+1;
            wf(m) = 0;
            for j = m : fdeg;
                wf(m) = wf(m) + rhnfhk(k,j-m+1,j-m+1,j-m+1)*x(var1+j);
            end;
            for j = 1 : cpts;
                wf(m) = wf(m) + x(var1+fdeg+j)*rhnfhk(k,fdeg+j-m+1,fdeg+j-m+1,fdeg-m+2);
            end;
        end;
        for m = 1 : gdeg + 1;
            wg(m) = 0;
            for j = m : gdeg;
                wg(m) = wg(m) + rhnfhk(k, j-m+1, j-m+1, j-m+1)*x(var2+j);
            end;
            for j = 1 : cpts;
                wg(m) = wg(m) + x(var2+gdeg+j)*rhnfhk(k,gdeg+j-m+1,gdeg+j-m+1,gdeg-m+2);
            end;
        end;
        fvec(eqn1+k) = epsilon*wf(5) + wf(4)*wf(1) + wg(2)*wg(1);
        fvec(eqn2+k) = epsilon*wg(3) + wf(1)*wg(2) - wf(2)*wg(1);
    end;
end;

%Set up the continuity equations
for i = 1: nint - 1;
    var1 = (i-1)*npi;
    eqn1 = var1 + bc + 2*cpts;
    var2 = var1 + fdeg + cpts;
    eqn2 = eqn1 + fdeg;
    for m = 1: fdeg;
        wf(m) = 0;
        for j = m: fdeg;
            wf(m) = wf(m) + rhnfhk(1,1,j-m+1,j-m+1)*x(var1+j);
        end;
        for j = 1: cpts;
            wf(m) = wf(m) + rhnfhk(1,1,fdeg+j-m+1,fdeg-m+2)*x(var1+fdeg+j);
        end;
    end;
    for m = 1: gdeg;
        wg(m) = 0;
        for j = m: gdeg;
            wg(m) = wg(m) + rhnfhk(1,1,j-m+1,j-m+1)*x(var2+j);
        end;
        for j = 1: cpts;
            wg(m) = wg(m) + rhnfhk(1,1,gdeg+j-m+1,gdeg-m+2)*x(var2+gdeg+j);
         end;
    end;
    for m = 1: fdeg;
        fvec(eqn1+m) = x(var1+npi+m) - wf(m);
    end;
    for m = 1: gdeg;
        fvec(eqn2+m) = x(var2+npi+m) - wg(m);
    end;
end;

%     Prepare for setting up the boundary conditions at t = 1.
var1 = n - npi;
for m = 1: fdeg + 1;
    wf(m) = 0;
    for j = m: fdeg;
        wf(m) = wf(m) + rhnfhk(1,1,j-m+1,j-m+1)*x(var1+j);
    end;
    for j = 1: cpts;
        wf(m) = wf(m) + rhnfhk(1,1,fdeg+j-m+1,fdeg-m+2)*x(var1+fdeg+j);
    end;
end;
var2 = var1 + fdeg + cpts;
for m = 1: gdeg + 1;
    wg(m) = 0;
    for j = m: gdeg;
        wg(m) = wg(m) + rhnfhk(1,1,j-m+1,j-m+1)*x(var2+j);
     end;
    for j = 1:cpts;
        wg(m) = wg(m) + rhnfhk(1,1,gdeg+j-m+1,gdeg-m+2)*x(var2+gdeg+j);
    end;
end;

%     Set up the boundary equations at t = 1.
%     f(1) = 0, f'(1) = 0, g(1) = omega2.
fvec(n-2) = wf(1);
fvec(n-1) = wf(2);
fvec(n) = wg(1) - omega2;
