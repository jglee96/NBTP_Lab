%%% 1D Distributed Bragg Reflctor with Additive Learning Algorithm %%%
clc; clear; close all;

% Setup program constants
Ngrid = single(200); % Number of permitivitty layers gird
dx = single(10);
Nstruct = single(5000); % Number of DBR structures
epsi = single(12.25); % layer permitivitty
eps0 = single(1); % air permitivitty
wQ = single(1); % weight of Q-factor
wMSL = single(0.07); % weight of Maximum Side Level

% Setup target lambda region
tarlam = single(300); % target wavelength
minlam = single(10); % minimum wavelength
maxlam = single(500); % maximum wavelength
lambda = linspace(minlam, maxlam, (maxlam-minlam)+1); % wavelength vector
tarlam_index = single(find(lambda==tarlam));

Layer = zeros(Nstruct,Ngrid,'single');
Layer(:,[1,end]) = 1;
rng('shuffle');
Layer(:,2:end-1) = round(rand(Nstruct,Ngrid-2,'single'));

% Initial DBR Calcultor
R = zeros(Nstruct,length(lambda),'single');
Q = zeros(Nstruct,1,'single');
MSL = zeros(Nstruct,1,'single');
parfor i=1:Nstruct
    R(i,:) = calR_single(Layer(i,:),lambda,Ngrid,dx,epsi,eps0);
    [Q(i), MSL(i)]= calQ_single(R(i,:),lambda,tarlam_index);
end
[Layer,R,Q,MSL] = DelDBR(Layer,R,Q,MSL);
Nstruct = single(length(Layer(:,1)));

% Learning Phase
E = zeros(1,Ngrid,'single');
for i = 1:Nstruct
    E = E+(wQ*Q(i)+(wMSL*MSL(i)))*Layer(i,:);
end

minE = min(E);
E = E-minE;
avgE = mean(E);

% Inference Phase
LayerF = zeros(1,Ngrid,'single');
for i=1:Ngrid
    if E(i) >=avgE
        LayerF(i) = 1;
    else
        LayerF(i) = 0;
    end
end
RF = calR_single(LayerF,lambda,Ngrid,dx,epsi,eps0);
[QF, MSLF] = calQ_single(RF,lambda,tarlam_index);
Plot_R(RF,lambda);
saveas(gcf,['Result Layers(Gird_' num2str(Ngrid) 'dx_' num2str(dx) ',Sample_' num2str(Nstruct) ',Wq_' num2str(wQ) ',Wmsl_' num2str(wMSL) ').jpg']);
save(['Result Layers(Gird_' num2str(Ngrid) 'dx_' num2str(dx) ',Sample_' num2str(Nstruct) ',Wq_' num2str(wQ) ',Wmsl_' num2str(wMSL) ').mat']);
