clear all
close all

x = 0:0.01:2*pi;

subplot(1,2,1);
f= -3./(5-4.*cos(x));
plot(x, f);
xlabel('radians');
title('R/b = 2');

subplot(1,2,2);
f= -15./(17-8.*cos(x));
plot(x, f);
xlabel('radians');
title('R/b = 4');


