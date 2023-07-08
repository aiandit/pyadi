function x = initialize_starting_point(nint)
% INITIALIZE_STARTING_POINT    Computes the standard starting point
%    for the swirling flow between two disks.
%
%    X = INITIALIZE_STARTING_POINT(NINT) computes the starting
%    point X whose length is 14*NINT where NINT is the number
%    of intervals used in the collocation method.

% STCS 2007 at RWTH Aachen University
% 04/20/07 by A. Vehreschild and H. Martin Buecker based on
%  the implementation of the MINPACK-2 routine DSFDFJ by
%  Brett M. Averick, 1993.

cpts = 4;
fdeg = 4;
gdeg = 2;
npi = 2*cpts+gdeg+fdeg;
omega1 = -1;
omega2 = 1;

n = 14*nint;
h = 1 / nint;
x = zeros(n,1);

adr = ((1:npi:n)-1)+fdeg+cpts;
x(adr+ 1) = omega1 + (omega2-omega1).*(0:nint-1).*h;
x(adr+ 2) = omega2 - omega1;

