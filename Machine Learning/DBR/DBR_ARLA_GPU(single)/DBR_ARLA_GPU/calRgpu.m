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
Pn = [epst, ones(Nstruct,1,'gpuArray')];
Pn1 = [ones(Nstruct,1,'gpuArray'), epst];
P = arrayfun(@(pn,pn1) sqrt(pn./pn1), Pn, Pn1);
kx = gpuArray(zeros(Nstruct,Ngrid+1,length(lambda)));
for l=1:length(lambda)
    kx(:,:,l) = sqrt(Pn)*2*pi/lambda(l);
end
h = [dx*ones(Nstruct,Ngrid,'gpuArray'), zeros(Nstruct,1,'gpuArray')];
B11 = gpuArray(zeros(Nstruct,Ngrid+1,length(lambda)));
B12 = gpuArray(zeros(Nstruct,Ngrid+1,length(lambda)));
B21 = gpuArray(zeros(Nstruct,Ngrid+1,length(lambda)));
B22 = gpuArray(zeros(Nstruct,Ngrid+1,length(lambda)));

for l=1:length(lambda)
    B11(:,:,l) = arrayfun(@(p,kx,h) (1+p).*exp(-1i.*kx.*h),P,kx(:,:,l),h);
    B12(:,:,l) = arrayfun(@(p,kx,h) (1-p).*exp(1i.*kx.*h),P,kx(:,:,l),h);
    B21(:,:,l) = arrayfun(@(p,kx,h) (1-p).*exp(-1i.*kx.*h),P,kx(:,:,l),h);
    B22(:,:,l) = arrayfun(@(p,kx,h) (1+p).*exp(1i.*kx.*h),P,kx(:,:,l),h);
end

B11 = gather(B11);
B12 = gather(B12);
B21 = gather(B21);
B22 = gather(B22);
R = zeros(Nstruct,length(lambda));

for i=1:Nstruct
    for l=1:length(lambda)
        Btot = eye(2);
        for j=1:Ngrid+1
            B = (1/2)*[B11(i,j,l) B12(i,j,l);B21(i,j,l) B22(i,j,l)];
            Btot = Btot*B;            
        end
        R(i,l) = abs(Btot(2,1)/Btot(1,1))^2;
    end
end
