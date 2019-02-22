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

B_tot = gpuArray(zeros(2,2,length(lambda)));
Pn = [epst, ones(Nstruct,1,'gpuArray')];
Pn1 = [ones(Nstruct,1,'gpuArray'), epst];
P = sqrt(Pn./Pn1);


% calculate transfer matrix (B_tot)
for nl = 1:Ngrid+1
    for l=1:length(lambda)
        
        Bf = @(P,kx,h) (1/2)*[(1+P)*exp(-1i*kx*h) (1-P)*exp(1i*kx*h); ...
            (1-P)*exp(-1i*kx*h) (1+P)*exp(1i*kx*h)];
        
        if nl == 1
            P = sqrt(epst(nl)/1);
            h = dx;
            kx = sqrt(epst(nl))*2*pi/(lambda(l));
            B_tot(:,:,l) = Bf(P,kx,h);
        elseif nl == Ngrid+1
            P = sqrt(1/epst(nl-1));
            h = 0;
            kx = 1*2*pi/(lambda(l));
            B_tot(:,:,l) = squeeze(B_tot(:,:,l))*Bf(P,kx,h);
        else
            P = sqrt(epst(nl)/epst(nl-1));
            h = dx;
            kx = sqrt(epst(nl))*2*pi/(lambda(l));
            B_tot(:,:,l) = squeeze(B_tot(:,:,l))*Bf(P,kx,h);
        end
    end
end
resultB = gather(B_tot);
R(1,:) = abs((resultB(2,1,:)./resultB(1,1,:))).^2;
