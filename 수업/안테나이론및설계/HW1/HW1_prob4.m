fa = cos(x);
fb = cos(x).^2;
fc = cos(x).^3;

fa = normFunc(fa);
fb = normFunc(fb);
fc = normFunc(fc);

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

function normf = normFunc(fx)
maxInt = max(fx);
normf = fx./maxInt;
end
