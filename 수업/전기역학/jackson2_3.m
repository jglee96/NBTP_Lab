clear all

x = linspace(0,5,100);

subplot(1,3,1);
x0=2; y0=1;
f= 1/pi.*(1./((x+x0).^2+y0.^2)-1./((x-x0).^2+y0.^2));
plot(x, f);
axis([0, 5, -0.3, 0]);
title('x0=2, y0=1');

subplot(1,3,2);
x0=1; y0=1;
f= 1/pi.*(1./((x+x0).^2+y0.^2)-1./((x-x0).^2+y0.^2));
plot(x, f);
axis([0, 5, -0.3, 0]);
title('x0=1, y0=1');

subplot(1,3,3);
x0=1; y0=2;
f= 1/pi.*(1./((x+x0).^2+y0.^2)-1./((x-x0).^2+y0.^2));
plot(x, f);
axis([0, 5, -0.3, 0]);
title('x0=1, y0=2');
