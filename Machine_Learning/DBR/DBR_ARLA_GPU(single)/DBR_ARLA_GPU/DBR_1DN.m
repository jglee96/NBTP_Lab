%%% 1D Distributed Bragg Reflctor with Additive Learning Algorithm %%%
clc; clear; close all;

% Setup program constants
Ngrid = 100; % Number of permitivitty layers gird
dx = 5;
Nstruct = 100; % Number of DBR structures
epsi = 12.25; % layer permitivitty
eps0 = 1; % air permitivitty
wQ = 1; % weight of Q-factor
wMSL = 0.06; % weight of Maximum Side Level
d = gpuDevice;

% Setup target lambda region
tarlam = 300; % target wavelength
minlam = 10; % minimum wavelength
maxlam = 500; % maximum wavelength
lambda = linspace(minlam, maxlam, (maxlam-minlam)+1); % wavelength vector
tarlam_index = find(lambda==tarlam);

Layer = zeros(Nstruct,Ngrid);
Layer(:,[1,end]) = 1;
rng('shuffle');
Layer(:,2:end-1) = round(rand(Nstruct,Ngrid-2));

% Initial DBR Calcultor
R = calRgpu(Layer,lambda,Nstruct,Ngrid,dx,epsi,eps0);
[Q, MSL]= calQ(R,lambda,Nstruct,tarlam_index);
[Layer,R,Q,MSL] = DelDBR(Layer,R,Q,MSL);
Nstruct = length(Layer(:,1));

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
Nstruct = 1;
RF = calRgpu(LayerF,lambda,Nstruct,Ngrid,dx,epsi,eps0);
[QF, MSLF] = calQ(RF,lambda,Nstruct,tarlam_index);
Plot_R(RF,lambda);
saveas(gcf,['Result Layers(Sample_' num2str(Nstruct) ',Wq_' num2str(wQ) ',Wmsl_' num2str(wMSL) ').jpg']);
save(['Result Parameters(Sample_' num2str(Nstruct) ',Wq_' num2str(wQ) ',Wmsl_' num2str(wMSL) ').mat']);
gpuDevice([]);
