[X,Y] = meshgrid(0:0.1:30,0:0.05:1);
Z1 = (nthroot(X,3).*(1-Y).^2)/3;
Z2 = X./(600*Y);
% s=surf(Z);
% s.EdgeColor='none';
% hold on;
figure(1)
contourf(X,Y,Z1);
xlabel('Q factor')
ylabel('Side Level Average')
% xticks(0:5:30)
% yticks(0:0.25:1)
colormap jet;
colorbar;

figure(2)
contourf(X,Y,Z2);
xlabel('Q factor')
ylabel('Side Level Average')
% xticks(0:5:30)
% yticks(0:0.25:1)
colormap jet;
colorbar;

