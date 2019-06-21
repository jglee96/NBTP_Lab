% [X,Y] = meshgrid(0:0.1:30,0:0.05:1);
% Z1 = (nthroot(X,3).*(1-Y))/3;
% % Z1 = ((X.^3).*(1-Y).^2)/3;
% % Z2 = X./(600*Y);
% [X3,Y3] = meshgrid(0:0.01:1,0:0.01:1);
% Z3 = X3./Y3;
% % s=surf(Z);
% % s.EdgeColor='none';
% % hold on;
% figure(1)
% contourf(X,Y,Z1);
% xlabel('Q factor')
% ylabel('Side Level Average')
% % xticks(0:5:30)
% % yticks(0:0.25:1)
% colormap jet;
% colorbar;
% 
% % figure(2)
% % contourf(X,Y,Z2);
% % xlabel('Q factor')
% % ylabel('Side Level Average')
% % % xticks(0:5:30)
% % % yticks(0:0.25:1)
% % colormap jet;
% % colorbar;
% 
% figure(3)
% contourf(X3,Y3,Z3);
% xlabel('Inside Range Average')
% ylabel('Outside Range Average')
% % xticks(0:5:30)
% % yticks(0:0.25:1)
% colormap jet;
% colorbar;

%% 2D splitter FOM %%
[X,Y] = meshgrid(0:0.01:2,0:0.01:2);
Z1 = 1 - ((X-1).^2 + (Y-1).^2);
Z2 = 1 - sqrt((X-1).^2 + (Y-1).^2);
figure(1);
contourf(X, Y, Z1);
xlabel('Output Port 1');
ylabel('Output Port 2');
colormap(jet);
colorbar;

figure(2);
contourf(X, Y, Z2);
xlabel('Output Port 1');
ylabel('Output Port 2');
colormap(jet);
colorbar;
