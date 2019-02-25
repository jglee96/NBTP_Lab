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
B_tot = gpuArray(eye(2,2));
Pn = [epst, ones(Nstruct,1,'gpuArray')];
Pn1 = [ones(Nstruct,1,'gpuArray'), epst];
P = arrayfun(@(pn,pn1) sqrt(pn./pn1), Pn, Pn1);
h = [dx*ones(Nstruct,Ngrid,'gpuArray'), zeros(Nstruct,1,'gpuArray')];
Bf = @(P,kx,h) (1/2)*[(1+P)*exp(-1i*kx*h) (1-P)*exp(1i*kx*h); ...
    (1-P)*exp(-1i*kx*h) (1+P)*exp(1i*kx*h)];
R = gpuArray(zeros(Nstruct,length(lambda)));

for l=1:length(lambda)
    B(:,:,l) = arrayfun(@(p,h) (1+p)*h,P,h);
end

% for i = 1:Nstruct
%     for l=1:length(lambda)
%         for j = 1:Ngrid+1
%             kx = sqrt(Pn(i,j))*2*pi/(lambda(l));
%             B_tot = B_tot*Bf(P(i,j),kx,h(j));
%         end
%         R(i,l) = abs((B_tot(2,1)./B_tot(1,1))).^2;
%     end
% end
% gather(R);
