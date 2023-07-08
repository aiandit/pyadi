N = 4

init = initialize_starting_point(N);
res = swirl(init, 1e-2);

save -ascii initm.txt init
save -ascii swirlm.txt res

system('cd .. && python3 -m swirl');

rho = load('rhnfhkm.txt');

pinit = load('../init.txt');
pswirl = load('../swirl.txt');
prho = load('../rhnfhk.txt');

erri = norm(init - pinit)
%[rho, prho]
errr = norm(rho - prho)
%[res, pswirl]
err = norm(res - pswirl)
