function R = calRgpu(Layer,lambda,Nstruct,Ngrid,dx,epsi,eps0)

warning off parallel:gpu:device:DeviceLibsNeedsRecompiling
try
    gpuArray.eye(2)^2;
catch ME
end
try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end

% DBR Calcultor
one_element = epsi*double((Layer == 1));
zero_element = eps0*double((Layer == 0));
epst = one_element + zero_element;
epst = gpuArray(epst);
lambda = gpuArray(lambda);

% calculate transfer matrix (B_tot)
B_tot = gpuArray(zeros(2,2,length(lambda)));
B_tot(1,1,:) = 1;
B_tot(2,2,:) = 1;
Pn = [epst, ones(Nstruct,1,'gpuArray')];
Pn1 = [ones(Nstruct,1,'gpuArray'), epst];
P = sqrt(Pn./Pn1);
h = [dx*ones(Nstruct,Ngrid,'gpuArray'), zeros(Nstruct,1,'gpuArray')];
Bf = @(P,kx,h) (1/2)*[(1+P)*exp(-1i*kx*h) (1-P)*exp(1i*kx*h); ...
    (1-P)*exp(-1i*kx*h) (1+P)*exp(1i*kx*h)];
R = zeros(Nstruct,length(lambda));

for i = 1:Nstruct
    for j = 1:Ngrid+1
        for l=1:length(lambda)
            kx = sqrt(Pn(i,j))*2*pi/(lambda(l));
            B_tot(:,:,l) = squeeze(B_tot(:,:,l))*Bf(P(i,j),kx,h(j));
        end
    end
    resultB = gather(B_tot);
    R(i,:) = abs((resultB(2,1,:)./resultB(1,1,:))).^2;
end
