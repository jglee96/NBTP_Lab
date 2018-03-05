clear all;
clc

syms k0;
syms d;
syms n;

T12 = (2*1)/(n+1);
T23 = (2*n)/(n+1);
R23 = (n-1)/(n+1);

% T1 = T12*T23*exp(-1i*k0*n*d);
% T2 = T12*R23^2*T23*exp(-1i*k0*n*d*3);
% T3 = T12*R23^4*T23*exp(-1i*k0*n*d*5);
% T_th = T1+T2+T3;

T_th = (4*n*exp(-1i*k0*n*d))/(((n+1)^2)-((n-1)^2)*exp(-2*1i*k0*n*d));

diff(T_th, n);


%% => (4*exp(-d*k_0*1i))/(n + 1)^2 - (8*n*exp(-d*k_0*1i))/(n + 1)^3
%% => (4*exp(-d*k_0*n*1i))/(n + 1)^2 - (8*n*exp(-d*k_0*n*1i))/(n + 1)^3 - (d*k_0*n*exp(-d*k_0*n*1i)*4i)/(n + 1)^2
%% => (4*exp(-d*k_0*n*1i))/(n + 1)^2 - (160*n^2*exp(-d*k_0*n*1i))/(n + 1)^6 + (2048*n^3*exp(-d*k_0*n*1i))/(n + 1)^9 - (4608*n^4*exp(-d*k_0*n*1i))/(n + 1)^10 - (8*n*exp(-d*k_0*n*1i))/(n + 1)^3 + (64*n*exp(-d*k_0*n*1i))/(n + 1)^5 - (d*k_0*n^2*exp(-d*k_0*n*1i)*32i)/(n + 1)^5 - (d*k_0*n^4*exp(-d*k_0*n*1i)*512i)/(n + 1)^9 - (d*k_0*n*exp(-d*k_0*n*1i)*4i)/(n + 1)^2
%% => (4*exp(-d*k0*n*1i))/(n + 1)^2 - (8*n*exp(-d*k0*n*1i))/(n + 1)^3 + (16*n*exp(-d*k0*n*1i)*(n - 1)^2)/(n + 1)^5 + (8*n^2*exp(-d*k0*n*1i)*(2*n - 2))/(n + 1)^5 - (40*n^2*exp(-d*k0*n*1i)*(n - 1)^2)/(n + 1)^6 + (128*n^3*exp(-d*k0*n*1i)*(n - 1)^4)/(n + 1)^9 + (128*n^4*exp(-d*k0*n*1i)*(n - 1)^3)/(n + 1)^9 - (288*n^4*exp(-d*k0*n*1i)*(n - 1)^4)/(n + 1)^10 - (d*k0*n*exp(-d*k0*n*1i)*4i)/(n + 1)^2 - (d*k0*n^2*exp(-d*k0*n*1i)*(n - 1)^2*8i)/(n + 1)^5 - (d*k0*n^4*exp(-d*k0*n*1i)*(n - 1)^4*32i)/(n + 1)^9
