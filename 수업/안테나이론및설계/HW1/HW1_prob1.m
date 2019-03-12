clc;clear;close all;

x = 0:0.05:2*pi;

fa = sin(x).^2;
fb = sin(x).*cos(x);
fc = (cos(x).^2).*sin(2*x);
fd = (sin(x).^2).*cos(x+pi/2);
fe = cos(x);

fa = normFunc(fa);
fb = normFunc(fb);
fc = normFunc(fc);
fd = normFunc(fd);
fe = normFunc(fe);

figure(1);
polarplot(x,fa);
hold on;
rlim([0 1]);
% polarplot(x,0.5*ones(length(x)));
pax = gca;
pax.ThetaAxisUnits = "radians";
hold off;

figure(2);
polarplot(x,fb);
hold on;
rlim([0 1]);
% polarplot(x,0.5*ones(length(x)));
pax = gca;
pax.ThetaAxisUnits = "radians";
hold off;

figure(3);
polarplot(x,fc);
hold on;
rlim([0 1]);
% polarplot(x,0.5*ones(length(x)));
pax = gca;
pax.ThetaAxisUnits = "radians";
hold off;

figure(4);
polarplot(x,fd);
hold on;
rlim([0 1]);
% polarplot(x,0.5*ones(length(x)));
pax = gca;
pax.ThetaAxisUnits = "radians";
hold off;

function normf = normFunc(fx)
maxInt = max(fx);
normf = fx./maxInt;
end
