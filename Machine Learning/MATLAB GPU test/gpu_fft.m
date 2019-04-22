clc;clear;close all;
n = 4*1e3;

A = rand(n,'single','gpuArray');
gd = gpuDevice();
tic;
B = fft(A);
wait(gd);
toc

AA = rand(n);
tic
BB = fft(AA);
toc;
