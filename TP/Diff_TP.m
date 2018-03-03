clear all;

syms k_0;
syms d;
syms n;

T12 = (2*1)/(n+1);
T23 = (2*n)/(n+1);

T_th = T12*T23*exp(-1i*k_0*n*d);


%% => (4*exp(-d*k_0*1i))/(n + 1)^2 - (8*n*exp(-d*k_0*1i))/(n + 1)^3
%% => (4*exp(-d*k_0*n*1i))/(n + 1)^2 - (8*n*exp(-d*k_0*n*1i))/(n + 1)^3 - (d*k_0*n*exp(-d*k_0*n*1i)*4i)/(n + 1)^2
