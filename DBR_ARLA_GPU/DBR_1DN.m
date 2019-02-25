%%% 1D Distributed Bragg Reflctor with Additive Learning Algorithm %%%
clc; clear; close all;

% Setup program constants
Ngrid = 50; % Number of permitivitty layers gird
dx = 10;
Nstruct = 500; % Number of DBR structures
epsi = 12.25; % layer permitivitty
eps0 = 1; % air permitivitty
wQ = 100; % weight of Q-factor
wMSL = 1; % weight of Maximum Side Level
d = gpuDevice;

% Setup target lambda region
tarlam = 800; % target wavelength
minlam = 400; % minimum wavelength
maxlam = 1200; % maximum wavelength
lambda = linspace(minlam, maxlam, (maxlam-minlam)+1); % wavelength vector
tarlam_index = find(lambda==tarlam);

Layer = zeros(Nstruct,Ngrid);
Layer(:,[1,end]) = 1;
Layer(:,2:end-1) = round(rand(Nstruct,Ngrid-2));

% Initial DBR Calcultor
R = calRgpu(Layer,lambda,Nstruct,Ngrid,dx,epsi,eps0);
[Q, MSL]= calQ(R,lambda,tarlam_index);
[Layer,R,Q,MSL] = DelDBR(Layer,R,Q,MSL);
Nstruct = length(Layer(:,1));
Plot_R(R,lambda);
saveas(gcf,['Initial Random Layers(' num2str(Nstruct) ')'],'jpg');
saveas(gcf,['Initial Random Layers(' num2str(Nstruct) ')'],'emf');

% Learning Phase
E = zeros(1,Ngrid);
for i = 1:Nstruct
    E = E+(wQ*Q(i)+(wMSL/MSL(i)))*Layer(i,:);
end

minE = min(E);
E = E-minE;
avgE = mean(E);

% Inference Phase
LayerF = zeros(1,Ngrid);
for i=1:Ngrid
    if E(i) >=avgE
        LayerF(i) = 1;
    else
        LayerF(i) = 0;
    end
end

RF = calRgpu(LayerF,lambda,Nstruct,Ngrid,dx,epsi,eps0);
[QF, MSLF] = calQ(RF, lambda, tarlam_index);
Plot_R(RF,lambda);
saveas(gcf,['Result Layers(' num2str(Nstruct) ')'],'jpg');
saveas(gcf,['Result Layers(' num2str(Nstruct) ')'],'emf');
gpuDevice([]);
